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

from paddle import fluid
import paddle.fluid.layers as L
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Constant, NumpyArrayInitializer
from ppdet.core.workspace import register, serializable

INF = 1e8
__all__ = ['SOLOLoss']


@register
@serializable
class SOLOLoss(object):
    def __init__(self,
                 batch_size=2,
                 loss_alpha=0.25,
                 loss_gamma=2.0,
                 clss_loss_weight=1.0,
                 ins_loss_weight=3.0):
        self.batch_size = batch_size
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma
        self.clss_loss_weight = clss_loss_weight
        self.ins_loss_weight = ins_loss_weight

    def dice_loss(self, pred_mask, gt_mask, gt_obj):
        a = L.reduce_sum(pred_mask * gt_mask, dim=[1, 2])
        b = L.reduce_sum(pred_mask * pred_mask, dim=[1, 2]) + 0.001
        c = L.reduce_sum(gt_mask * gt_mask, dim=[1, 2]) + 0.001
        d = (2 * a) / (b + c)
        loss_mask_mask = L.reshape(gt_obj, (-1,))  # 掩码损失的掩码。
        return (1 - d) * loss_mask_mask

    def __call__(self, kernel_preds, cls_preds, mask_protos, batch_gt_objs_tensors,
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

        batch_size = self.batch_size
        num_layers = len(kernel_preds)

        # ================= 计算损失 =================
        num_ins = 0.  # 记录这一批图片的正样本个数
        loss_clss, loss_masks = [], []
        for bid in range(batch_size):
            for lid in range(num_layers):
                # ================ 掩码损失 ======================
                mask_proto = mask_protos[bid]                           # [256, s4, s4]   这张图片产生的掩码原型。
                kernel_pred = kernel_preds[lid][bid]                    # [256, seg_num_grid, seg_num_grid]   格子预测的卷积核（yolact中的“掩码系数”）
                kernel_pred = L.transpose(kernel_pred, perm=[1, 2, 0])  # [seg_num_grid, seg_num_grid, 256]   格子预测的卷积核（yolact中的“掩码系数”）

                gt_objs = batch_gt_objs_tensors[lid][bid]    # [seg_num_grid, seg_num_grid, 1]
                gt_masks = batch_gt_masks_tensors[lid][bid]  # [-1, s4, s4]
                pmidx = batch_gt_pos_idx_tensors[lid][bid]   # [-1, 3]
                gt_objs.stop_gradient = True
                gt_masks.stop_gradient = True
                pmidx.stop_gradient = True

                idx_sum = L.reduce_sum(pmidx, dim=1)
                keep = L.where(idx_sum > -1)
                keep = L.reshape(keep, (-1,))
                keep.stop_gradient = True
                pmidx = L.gather(pmidx, keep)  # [M, 3]

                yx_idx = pmidx[:, :2]  # [M, 2]
                m_idx = pmidx[:, 2]    # [M, ]
                yx_idx.stop_gradient = True
                m_idx.stop_gradient = True

                # 抽出来
                gt_obj = L.gather_nd(gt_objs, yx_idx)         # [M, 1]        是否是真正的正样本。
                pos_krn = L.gather_nd(kernel_pred, yx_idx)    # [M, 256]      正样本的卷积核（掩码系数）。
                gt_mask = L.gather(gt_masks, m_idx)           # [M, s4, s4]   真实掩码。

                # 正样本数量
                num_ins += L.reduce_sum(gt_obj)

                # 生成预测掩码
                mask_proto = L.transpose(mask_proto, perm=[1, 2, 0])  # [s4, s4, 256]
                masks = L.matmul(mask_proto, pos_krn, transpose_y=True)  # [s4, s4, M]
                masks = L.sigmoid(masks)  # [s4, s4, M]
                masks = L.transpose(masks, perm=[2, 0, 1])  # [M, s4, s4]
                loss_mask = self.dice_loss(masks, gt_mask, gt_obj)
                loss_masks.append(loss_mask)

                # ================ 分类损失。sigmoid_focal_loss() ======================
                gamma = self.loss_gamma
                alpha = self.loss_alpha
                pred_conf = cls_preds[lid][bid]                       # [80, seg_num_grid, seg_num_grid]    未进行sigmoid()激活。
                pred_conf = L.transpose(pred_conf, perm=[1, 2, 0])    # [seg_num_grid, seg_num_grid, 80]    未进行sigmoid()激活。
                pred_conf = L.sigmoid(pred_conf)                      # [seg_num_grid, seg_num_grid, 80]    已进行sigmoid()激活。
                gt_clss = batch_gt_clss_tensors[lid][bid]             # [seg_num_grid, seg_num_grid, 80]    真实类别onehot
                gt_clss.stop_gradient = True
                pos_loss = gt_clss * (0 - L.log(pred_conf + 1e-9)) * L.pow(1 - pred_conf, gamma) * alpha
                neg_loss = (1.0 - gt_clss) * (0 - L.log(1 - pred_conf + 1e-9)) * L.pow(pred_conf, gamma) * (1 - alpha)
                focal_loss = pos_loss + neg_loss
                focal_loss = L.reduce_sum(focal_loss, dim=[0, 1])
                loss_clss.append(focal_loss)
        loss_masks = L.concat(loss_masks, axis=0)
        loss_masks = L.reduce_sum(loss_masks) * self.ins_loss_weight
        loss_masks = loss_masks / L.elementwise_max(L.ones((1,), dtype='float32'), num_ins)

        loss_clss = L.concat(loss_clss, axis=0)
        loss_clss = L.reduce_sum(loss_clss) * self.clss_loss_weight
        loss_clss = loss_clss / L.elementwise_max(L.ones((1,), dtype='float32'), num_ins)

        loss_all = {
            "loss_masks": loss_masks,
            "loss_clss": loss_clss
        }
        return loss_all


