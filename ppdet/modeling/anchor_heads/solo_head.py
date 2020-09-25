# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Constant, NumpyArrayInitializer
from paddle.fluid.regularizer import L2Decay
from ppdet.modeling.ops import ConvNorm, DeformConvNorm, Conv2dUnit, Upsample
from ppdet.modeling.ops import MultiClassNMS
import paddle.fluid.layers as L

from ppdet.core.workspace import register



def concat_coord(x):
    ins_feat = x  # [N, c, h, w]

    batch_size = L.shape(x)[0]
    h = L.shape(x)[2]
    w = L.shape(x)[3]
    float_h = L.cast(h, 'float32')
    float_w = L.cast(w, 'float32')

    y_range = L.range(0., float_h, 1., dtype='float32')     # [h, ]
    y_range = 2.0 * y_range / (float_h - 1.0) - 1.0
    x_range = L.range(0., float_w, 1., dtype='float32')     # [w, ]
    x_range = 2.0 * x_range / (float_w - 1.0) - 1.0
    x_range = L.reshape(x_range, (1, -1))   # [1, w]
    y_range = L.reshape(y_range, (-1, 1))   # [h, 1]
    x = L.expand(x_range, [h, 1])     # [h, w]
    y = L.expand(y_range, [1, w])     # [h, w]

    x = L.reshape(x, (1, 1, h, w))   # [1, 1, h, w]
    y = L.reshape(y, (1, 1, h, w))   # [1, 1, h, w]
    x = L.expand(x, [batch_size, 1, 1, 1])   # [N, 1, h, w]
    y = L.expand(y, [batch_size, 1, 1, 1])   # [N, 1, h, w]

    ins_kernel_feat = L.concat([ins_feat, x, y], axis=1)   # [N, c+2, h, w]

    return ins_kernel_feat

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = L.pool2d(heat, pool_size=kernel, pool_stride=1,
                    pool_padding=[[0, 0], [0, 0], [1, 0], [1, 0]],
                    pool_type='max')
    keep = L.cast(L.equal(hmax, heat), 'float32')
    return heat * keep


def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)   0、1组成的掩码
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss'
        sigma (float): std in gaussian method
        sum_masks (Tensor):  shape (n, )      n个物体的面积

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = L.shape(cate_labels)[0]   # 物体数
    seg_masks = L.reshape(seg_masks, (n_samples, -1))   # [n, h*w]
    # inter.
    inter_matrix = L.matmul(seg_masks, seg_masks, transpose_y=True)   # [n, n] 自己乘以自己的转置。两两之间的交集面积。
    # union.
    sum_masks_x = L.expand(L.reshape(sum_masks, (1, -1)), [n_samples, 1])     # [n, n]  sum_masks重复了n行得到sum_masks_x
    # iou.
    iou_matrix = inter_matrix / (sum_masks_x + L.transpose(sum_masks_x, [1, 0]) - inter_matrix)
    rows = L.range(0, n_samples, 1, 'int32')
    cols = L.range(0, n_samples, 1, 'int32')
    rows = L.expand(L.reshape(rows, (1, -1)), [n_samples, 1])
    cols = L.expand(L.reshape(cols, (-1, 1)), [1, n_samples])
    tri_mask = L.cast(rows > cols, 'float32')
    iou_matrix = tri_mask * iou_matrix   # [n, n]   只取上三角部分

    # label_specific matrix.
    cate_labels_x = L.expand(L.reshape(cate_labels, (1, -1)), [n_samples, 1])     # [n, n]  cate_labels重复了n行得到cate_labels_x
    label_matrix = L.cast(L.equal(cate_labels_x, L.transpose(cate_labels_x, [1, 0])), 'float32')
    label_matrix = tri_mask * label_matrix   # [n, n]   只取上三角部分

    # IoU compensation
    compensate_iou = L.reduce_max(iou_matrix * label_matrix, dim=0)
    compensate_iou = L.expand(L.reshape(compensate_iou, (1, -1)), [n_samples, 1])     # [n, n]
    compensate_iou = L.transpose(compensate_iou, [1, 0])      # [n, n]

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # # matrix nms
    if kernel == 'gaussian':
        decay_matrix = L.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = L.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient = L.reduce_min((decay_matrix / compensate_matrix), dim=0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient = L.reduce_min(decay_matrix, dim=0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update



__all__ = ['SOLOv2Head', 'MaskFeatHead']


@register
class SOLOv2Head(object):
    __inject__ = ['solo_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=81,
                 in_channels=256,
                 seg_feat_channels=512,
                 prior_prob=0.01,
                 num_convs=4,
                 strides=[8, 8, 16, 32, 32],
                 kernel_out_channels=256,
                 sigma=0.2,
                 num_grids=None,
                 solo_loss=None,
                 nms_cfg=None):
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.strides = strides
        self.sigma = sigma

        self.prior_prob = prior_prob
        self.num_convs = num_convs
        self.seg_feat_channels = seg_feat_channels
        self.kernel_out_channels = kernel_out_channels
        self.solo_loss = solo_loss
        self.nms_cfg = nms_cfg
        self._init_layers()

    def _init_layers(self):
        self.cls_convs = []   # 每个fpn输出特征图  共享的  再进行卷积的卷积层，用于预测类别
        self.krn_convs = []   # 每个fpn输出特征图  共享的  再进行卷积的卷积层，用于预测卷积核

        # 每个fpn输出特征图  共享的  卷积层。
        for lvl in range(0, self.num_convs):
            # 使用gn，组数是32，而且带激活relu
            in_ch = self.in_channels if lvl == 0 else self.seg_feat_channels
            cls_conv_layer = Conv2dUnit(in_ch, self.seg_feat_channels, 3, stride=1, bias_attr=False, gn=1, groups=32, act='relu', name='head.cls_convs.%d' % (lvl, ))
            self.cls_convs.append(cls_conv_layer)

            in_ch = self.in_channels + 2 if lvl == 0 else self.seg_feat_channels
            krn_conv_layer = Conv2dUnit(in_ch, self.seg_feat_channels, 3, stride=1, bias_attr=False, gn=1, groups=32, act='relu', name='head.krn_convs.%d' % (lvl, ))
            self.krn_convs.append(krn_conv_layer)
        # 类别分支最后的卷积。设置偏移的初始值使得各类别预测概率初始值为self.prior_prob (根据激活函数是sigmoid()时推导出，和RetinaNet中一样)
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        cls_last_conv_layer = Conv2dUnit(self.seg_feat_channels, self.cate_out_channels, 3, stride=1, bias_attr=True, act=None, bias_init_value=bias_init_value, name='head.cls_convs.%d' % (self.num_convs, ))
        # 卷积核分支最后的卷积
        krn_last_conv_layer = Conv2dUnit(self.seg_feat_channels, self.kernel_out_channels, 3, stride=1, bias_attr=True, act=None, name='head.krn_convs.%d' % (self.num_convs, ))
        self.cls_convs.append(cls_last_conv_layer)
        self.krn_convs.append(krn_last_conv_layer)

    def get_prediction(self, feats, eval=True):
        name_list = list(feats.keys())
        feats2 = [feats[name] for name in name_list]   # [p2, p3, p4, p5]
        feats = feats2
        # 有5个张量，5个张量的strides=[8, 8, 16, 32, 32]，所以先对首尾张量进行插值。
        # 一定要设置align_corners=False, align_mode=0才能和原版SOLO输出一致。
        new_feats = [L.resize_bilinear(feats[0], out_shape=L.shape(feats[1])[2:], align_corners=False, align_mode=0),
                     feats[1],
                     feats[2],
                     feats[3],
                     L.resize_bilinear(feats[4], out_shape=L.shape(feats[3])[2:], align_corners=False, align_mode=0)]

        kernel_preds, cls_preds = [], []
        for idx in range(len(self.seg_num_grids)):
            krn_feat = new_feats[idx]   # 给卷积核分支

            # ============ kernel branch (卷积核分支) ============
            ins_kernel_feat = concat_coord(krn_feat)   # 带上坐标信息。[N, c+2, h, w]
            kernel_feat = ins_kernel_feat      # ins_kernel_feat不再使用
            seg_num_grid = self.seg_num_grids[idx]   # 这个特征图一行(列)的格子数
            # kernel_feat插值成格子图。 [N, c+2, seg_num_grid, seg_num_grid]
            kernel_feat = L.resize_bilinear(kernel_feat, out_shape=[seg_num_grid, seg_num_grid], align_corners=False, align_mode=0)

            # 扔掉插入的坐标那2个通道，作为cls_feat。 [N, c, seg_num_grid, seg_num_grid]
            cls_feat = kernel_feat[:, :-2, :, :]

            for kernel_layer in self.krn_convs:
                kernel_feat = kernel_layer(kernel_feat)
            for class_layer in self.cls_convs:
                cls_feat = class_layer(cls_feat)
            kernel_pred = kernel_feat   # [N, 256, seg_num_grid, seg_num_grid]   每个格子的预测卷积核
            cls_pred = cls_feat         # [N,  80, seg_num_grid, seg_num_grid]   每个格子的预测概率，未进行sigmoid()激活

            if eval:
                # [N, seg_num_grid, seg_num_grid, 80]   每个格子的预测概率，已进行sigmoid()激活
                cls_pred = L.transpose(points_nms(L.sigmoid(cls_pred), kernel=2), perm=[0, 2, 3, 1])

            kernel_preds.append(kernel_pred)
            cls_preds.append(cls_pred)
        return [kernel_preds, cls_preds]

    def get_seg(self, kernel_preds, cls_preds, mask_protos, ori_shapes, resize_shapes):
        num_levels = len(cls_preds)   # 输出层个数=5
        featmap_size = L.shape(mask_protos)[-2:]   # 特征图大小，为stride=4

        result_list = []
        # for img_id in range(len(img_metas)):
        for img_id in range(1):
            cate_pred_list = [
                L.reshape(cls_preds[i][img_id], (-1, self.cate_out_channels)) for i in range(num_levels)
            ]
            mask_proto = mask_protos[img_id:img_id + 1, :, :, :]
            kernel_pred_list = [
                L.reshape(L.transpose(kernel_preds[i][img_id], perm=[1, 2, 0]), (-1, self.kernel_out_channels)) for i in range(num_levels)
            ]
            resize_shape = resize_shapes[img_id]
            ori_shape = ori_shapes[img_id]

            cate_pred_list = L.concat(cate_pred_list, axis=0)
            kernel_pred_list = L.concat(kernel_pred_list, axis=0)

            masks, classes, scores = self.get_seg_single(cate_pred_list, mask_proto, kernel_pred_list,
                                         featmap_size, resize_shape, ori_shape)
        #     result_list.append(result)
        # return result_list
        return {'masks': masks, 'classes': classes, 'scores': scores, }


    def get_seg_single(self,
                       cate_preds,
                       mask_proto,
                       kernel_preds,
                       featmap_size,
                       resize_shape,
                       ori_shape):
        '''

        :param cate_preds:   [所有格子数, 80]
        :param mask_proto:   [1, 256, s4, s4]   掩码原型
        :param kernel_preds:   [所有格子数, 256]   每个格子生成的卷积核，是1x1卷积核，输入通道数是256，即掩码原型的通道数。
        :param featmap_size:   (s4, s4)
        :param resize_shape:   shape=[3, ]
        :param ori_shape:      shape=[3, ]
        :return:
        '''
        # overall info.
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)   # 输入网络的图片大小
        cfg = self.nms_cfg

        # 第一次过滤，分数过滤
        inds = L.where(cate_preds > cfg['score_thr'])   # [M, 2]
        # if len(inds) == 0:
        #     return None
        # 静态图里写条件判断太难了。
        def exist_objs_1(inds, cate_preds):
            inds.stop_gradient = True
            scores = L.gather_nd(cate_preds, inds)  # [M, ]   M个物体的分数
            return inds, scores
        def no_objs_1(cate_preds):
            inds = L.zeros((1, 2), np.int64)
            inds.stop_gradient = True
            scores = L.gather_nd(cate_preds, inds) - 99.0   # [M, ]   M个物体的分数。后面会被过滤掉。
            return inds, scores
        # 是否有物体
        inds, scores = L.cond(L.shape(inds)[0] == 0, lambda: no_objs_1(cate_preds), lambda: exist_objs_1(inds, cate_preds))


        classes = inds[:, 1]   # [M, ]   M个物体的类别id
        kernel_preds = L.gather(kernel_preds, inds[:, 0])   # [M, 256]   M个物体的卷积核


        n_stage = len(self.seg_num_grids)   # 5个输出层
        strides = []
        for ind_ in range(n_stage):
            st = L.zeros((1, ), dtype=np.float32) + self.strides[ind_]
            st = L.expand(st, [self.seg_num_grids[ind_] ** 2, ])  # [40*40, ]
            strides.append(st)
        strides = L.concat(strides, axis=0)
        strides.stop_gradient = True
        strides = L.gather(strides, inds[:, 0])   # [M, ]   M个物体的下采样倍率

        # mask encoding.原版SOLO中的写法。1x1的卷积核卷积掩码原型，即可得到掩码。
        # M, C = kernel_preds.shape
        # kernel_preds = kernel_preds.view(M, C, 1, 1)    # 被当做卷积核使
        # seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()
        # 1x1的卷积核卷积掩码原型，等价于矩阵相乘。注意，3x3卷积核的话可不等价。
        # 这里是由于暂时没发现等价api，所以用矩阵相乘代替。solov2和yolact在这里是一样的。
        mask_proto = L.squeeze(mask_proto, axes=[0])   # [256, s4, s4]
        mask_proto = L.transpose(mask_proto, perm=[1, 2, 0])   # [s4, s4, 256]
        masks = L.matmul(mask_proto, kernel_preds, transpose_y=True)   # [s4, s4, M]
        masks = L.sigmoid(masks)   # [s4, s4, M]
        masks = L.transpose(masks, perm=[2, 0, 1])   # [M, s4, s4]

        # mask.
        seg_masks = L.cast(masks > cfg['mask_thr'], 'float32')   # [M, s4, s4]   前景的话值为1
        sum_masks = L.reduce_sum(seg_masks, dim=[1, 2])   # [M, ]   M个物体的掩码面积


        # 第二次过滤，下采样倍率过滤。掩码的面积 超过 下采样倍率 才保留下来。
        keep = L.where(sum_masks > strides)
        # if keep.sum() == 0:
        #     return None

        # 静态图里写条件判断太难了。
        def exist_objs_2(keep, seg_masks, masks, sum_masks, scores, classes):
            keep = L.reshape(keep, (-1,))  # [M2, ]
            keep.stop_gradient = True
            seg_masks = L.gather(seg_masks, keep)  # [M2, s4, s4]   M2个物体的掩码
            masks = L.gather(masks, keep)          # [M2, s4, s4]   M2个物体的掩码概率
            sum_masks = L.gather(sum_masks, keep)  # [M2, ]   M2个物体的掩码面积
            scores = L.gather(scores, keep)        # [M2, ]   M2个物体的分数
            classes = L.gather(classes, keep)      # [M2, ]   M2个物体的类别id
            return seg_masks, masks, sum_masks, scores, classes

        def no_objs_2(seg_masks, masks, sum_masks, scores, classes):
            keep = L.zeros((1, ), np.int64)
            keep.stop_gradient = True
            seg_masks = L.gather(seg_masks, keep)   # [M2, s4, s4]   M2个物体的掩码
            masks = L.gather(masks, keep)           # [M2, s4, s4]   M2个物体的掩码概率
            sum_masks = L.gather(sum_masks, keep)   # [M2, ]   M2个物体的掩码面积
            scores = L.gather(scores, keep) - 99.0  # [M2, ]   M2个物体的分数。负分数，后面会被过滤掉。
            classes = L.gather(classes, keep)       # [M2, ]   M2个物体的类别id
            return seg_masks, masks, sum_masks, scores, classes

        # 是否有物体
        seg_masks, masks, sum_masks, scores, classes = L.cond(L.shape(keep)[0] == 0,
                                        lambda: no_objs_2(seg_masks, masks, sum_masks, scores, classes),
                                        lambda: exist_objs_2(keep, seg_masks, masks, sum_masks, scores, classes))



        # mask scoring.
        # [M2, ]   前景的掩码概率求和，再除以掩码面积。即M2个物体的前景部分的平均掩码概率
        avg_prob = L.reduce_sum(masks * seg_masks, dim=[1, 2]) / sum_masks
        scores *= avg_prob   # [M2, ]   M2个物体的最终分数 = 分类概率 * 平均掩码概率

        # 第三次过滤，只保留得分前cfg['nms_pre']个物体
        _, sort_inds = L.argsort(scores, axis=-1, descending=True)   # 最终分数降序。最大值的下标，第2大值的下标，...
        sort_inds = sort_inds[:cfg['nms_pre']]   # 最多cfg['nms_pre']个物体。

        seg_masks = L.gather(seg_masks, sort_inds)   # [M3, s4, s4]   M3个物体的掩码
        masks = L.gather(masks, sort_inds)           # [M3, s4, s4]   M3个物体的掩码概率
        sum_masks = L.gather(sum_masks, sort_inds)   # [M3, ]   M3个物体的掩码面积
        scores = L.gather(scores, sort_inds)         # [M3, ]   M3个物体的分数
        classes = L.gather(classes, sort_inds)       # [M3, ]   M3个物体的类别id

        # Matrix NMS
        scores = matrix_nms(seg_masks, classes, scores,
                                 kernel=cfg['kernel'], sigma=cfg['sigma'], sum_masks=sum_masks)

        # 第四次过滤，分数过滤
        keep = L.where(scores >= cfg['update_thr'])
        # if keep.sum() == 0:
        #     return None

        def exist_objs_3(keep, masks, classes, scores, upsampled_size_out, resize_shape, ori_shape):
            keep = L.reshape(keep, (-1,))
            keep.stop_gradient = True
            masks = L.gather(masks, keep)      # [M4, s4, s4]   M4个物体的掩码概率
            scores = L.gather(scores, keep)    # [M4, ]   M4个物体的分数
            classes = L.gather(classes, keep)  # [M4, ]   M4个物体的类别id

            # 第五次过滤，只保留得分前cfg['max_per_img']个物体
            _, sort_inds = L.argsort(scores, axis=-1, descending=True)
            sort_inds = sort_inds[:cfg['max_per_img']]
            sort_inds.stop_gradient = True

            masks = L.gather(masks, sort_inds)      # [M5, s4, s4]   M5个物体的掩码概率
            scores = L.gather(scores, sort_inds)    # [M5, ]   M5个物体的分数
            classes = L.gather(classes, sort_inds)  # [M5, ]   M5个物体的类别id

            masks = L.resize_bilinear(L.unsqueeze(masks, axes=[0]), out_shape=upsampled_size_out,
                                      align_corners=False, align_mode=0)[:, :, :resize_shape[0], :resize_shape[1]]  # 去掉黑边
            masks = L.resize_bilinear(masks, out_shape=ori_shape[:2],
                                      align_corners=False, align_mode=0)  # 插值成原图大小
            masks = L.cast(masks > cfg['mask_thr'], 'float32')[0]
            return masks, classes, scores

        def no_objs_3():
            masks = L.zeros([1, 1, 1], 'float32') - 1.0
            classes = L.zeros([1, ], 'int64') - 1
            scores = L.zeros([1, ], 'float32') - 2.0
            return masks, classes, scores

        # 是否有物体
        masks, classes, scores = L.cond(L.shape(keep)[0] == 0,
                                        no_objs_3,
                                        lambda: exist_objs_3(keep, masks, classes, scores, upsampled_size_out, resize_shape, ori_shape))
        return masks, classes, scores


    def get_loss(self, kernel_preds, cls_preds, mask_protos, batch_gt_objs_tensors,
                 batch_gt_clss_tensors, batch_gt_masks_tensors, batch_gt_pos_idx_tensors):
        '''
        :param kernel_preds:  kernel_preds里每个元素形状是[N, 256, seg_num_grid, seg_num_grid],  每个格子的预测卷积核。      从 小感受野 到 大感受野。
        :param cls_preds:     cls_preds里每个元素形状是   [N,  80, seg_num_grid, seg_num_grid],  每个格子的预测概率，未进行sigmoid()激活。  从 小感受野 到 大感受野。
        :param mask_protos:   [bs, 256, s4, s4]   掩码原型
        :param batch_gt_objs_tensors:   里每个元素形状是[N, seg_num_grid, seg_num_grid, 1],   每个格子的objness。           从 小感受野 到 大感受野。
        :param batch_gt_clss_tensors:   里每个元素形状是[N, seg_num_grid, seg_num_grid, 80],  每个格子真实类别onehot。      从 小感受野 到 大感受野。
        :param batch_gt_masks_tensors:     里每个元素形状是[N, -1, s4, s4],   真实掩码。  从 小感受野 到 大感受野。
        :param batch_gt_pos_idx_tensors:   里每个元素形状是[N, -1, 3],    正样本的下标。  从 小感受野 到 大感受野。
        :return:
        '''
        loss = self.solo_loss(kernel_preds, cls_preds, mask_protos, batch_gt_objs_tensors,
                              batch_gt_clss_tensors, batch_gt_masks_tensors, batch_gt_pos_idx_tensors)
        return loss




@register
class MaskFeatHead(object):
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level,
                 end_level,
                 num_classes):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.num_classes = num_classes

        self.convs_all_levels = []
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = []
            if i == 0:
                one_conv = Conv2dUnit(self.in_channels, self.out_channels, 3, stride=1, bias_attr=False, gn=1,
                                            groups=32, act='relu', name='mask_feat_head.convs_all_levels.%d.conv0' % (i,))
                convs_per_level.append(one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.in_channels+2 if i==3 else self.in_channels
                    one_conv = Conv2dUnit(chn, self.out_channels, 3, stride=1, bias_attr=False, gn=1,
                                                groups=32, act='relu', name='mask_feat_head.convs_all_levels.%d.conv%d' % (i, j))
                    convs_per_level.append(one_conv)
                    one_upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.append(one_upsample)
                    continue

                one_conv = Conv2dUnit(self.out_channels, self.out_channels, 3, stride=1, bias_attr=False, gn=1,
                                      groups=32, act='relu', name='mask_feat_head.convs_all_levels.%d.conv%d' % (i, j))
                convs_per_level.append(one_conv)
                one_upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.append(one_upsample)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = Conv2dUnit(self.out_channels, self.num_classes, 1, stride=1, bias_attr=False, gn=1,
                              groups=32, act='relu', name='mask_feat_head.conv_pred.0')

    def get_mask_feats(self, inputs):
        name_list = list(inputs.keys())
        name_list = name_list[self.start_level:self.end_level+1]   # [p2, p3, p4, p5]
        inputs2 = [inputs[name] for name in name_list]   # [p2, p3, p4, p5]
        inputs = inputs2

        feature_add_all_level = self.convs_all_levels[0][0](inputs[0])

        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == 3:
                input_feat = input_p
                batch_size = L.shape(input_feat)[0]
                h = L.shape(input_feat)[2]
                w = L.shape(input_feat)[3]
                float_h = L.cast(h, 'float32')
                float_w = L.cast(w, 'float32')

                y_range = L.range(0., float_h, 1., dtype='float32')  # [h, ]
                y_range = 2.0 * y_range / (float_h - 1.0) - 1.0
                x_range = L.range(0., float_w, 1., dtype='float32')  # [w, ]
                x_range = 2.0 * x_range / (float_w - 1.0) - 1.0
                x_range = L.reshape(x_range, (1, -1))  # [1, w]
                y_range = L.reshape(y_range, (-1, 1))  # [h, 1]
                x = L.expand(x_range, [h, 1])  # [h, w]
                y = L.expand(y_range, [1, w])  # [h, w]

                x = L.reshape(x, (1, 1, h, w))  # [1, 1, h, w]
                y = L.reshape(y, (1, 1, h, w))  # [1, 1, h, w]
                x = L.expand(x, [batch_size, 1, 1, 1])  # [N, 1, h, w]
                y = L.expand(y, [batch_size, 1, 1, 1])  # [N, 1, h, w]

                input_p = L.concat([input_p, x, y], axis=1)  # [N, c+2, h, w]

            for ly in self.convs_all_levels[i]:
                input_p = ly(input_p)
            feature_add_all_level += input_p

        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred










