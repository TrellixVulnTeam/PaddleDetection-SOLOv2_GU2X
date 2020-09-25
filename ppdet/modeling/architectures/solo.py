# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict

import paddle.fluid as fluid

from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register

__all__ = ['SOLOv2']


@register
class SOLOv2(object):
    __category__ = 'architecture'
    __inject__ = ['backbone', 'fpn', 'mask_feat_head', 'solo_head']

    def __init__(self, backbone, fpn, mask_feat_head, solo_head):
        super(SOLOv2, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.mask_feat_head = mask_feat_head
        self.solo_head = solo_head

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']

        mixed_precision_enabled = mixed_precision_global_state() is not None
        # cast inputs to FP16
        if mixed_precision_enabled:
            im = fluid.layers.cast(im, 'float16')

        # backbone
        body_feats = self.backbone(im)

        # cast features back to FP32
        if mixed_precision_enabled:
            body_feats = OrderedDict((k, fluid.layers.cast(v, 'float32'))
                                     for k, v in body_feats.items())

        # FPN
        body_feats, spatial_scale = self.fpn.get_output(body_feats)

        # MaskFeatHead。 [bs, 256, s4, s4]   掩码原型
        mask_feats = self.mask_feat_head.get_mask_feats(body_feats)

        # SOLOv2Head
        if mode == 'train':
            # kernel_preds里每个元素形状是[N, 256, seg_num_grid, seg_num_grid],  每个格子的预测卷积核。      从 小感受野 到 大感受野。
            # cls_preds里每个元素形状是   [N,  80, seg_num_grid, seg_num_grid],  每个格子的预测概率，未进行sigmoid()激活。  从 小感受野 到 大感受野。
            kernel_preds, cls_preds = self.solo_head.get_prediction(body_feats, eval=False)
            gt_objs = []
            gt_clss = []
            gt_masks = []
            gt_pos_idx = []
            for i in range(len(self.solo_head.strides)):
                gt_objs.append(feed_vars['layer%d_gt_objs' % i])
                gt_clss.append(feed_vars['layer%d_gt_clss' % i])
                gt_masks.append(feed_vars['layer%d_gt_masks' % i])
                gt_pos_idx.append(feed_vars['layer%d_gt_pos_idx' % i])
            loss = self.solo_head.get_loss(kernel_preds, cls_preds, mask_feats, gt_objs, gt_clss, gt_masks, gt_pos_idx)
            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            ori_shape = feed_vars['ori_shape']
            resize_shape = feed_vars['resize_shape']

            # kernel_preds里每个元素形状是[N, 256, seg_num_grid, seg_num_grid],  每个格子的预测卷积核。      从 小感受野 到 大感受野。
            # cls_preds里每个元素形状是   [N, seg_num_grid, seg_num_grid,  80],  每个格子的预测概率，已进行sigmoid()激活。  从 小感受野 到 大感受野。
            kernel_preds, cls_preds = self.solo_head.get_prediction(body_feats, eval=True)
            pred = self.solo_head.get_seg(kernel_preds, cls_preds, mask_feats, ori_shape, resize_shape)
            return pred

    def _inputs_def(self, image_shape, fields):
        im_shape = [None] + image_shape
        # yapf: disable
        inputs_def = {
            'image':    {'shape': im_shape,  'dtype': 'float32', 'lod_level': 0},
            'ori_shape': {'shape': [None, 3], 'dtype': 'int32', 'lod_level': 0},
            'resize_shape': {'shape': [None, 3], 'dtype': 'int32', 'lod_level': 0},
            'im_info':  {'shape': [None, 3], 'dtype': 'float32', 'lod_level': 0},
            'im_id':    {'shape': [None, 1], 'dtype': 'int64',   'lod_level': 0},
            'gt_bbox':  {'shape': [None, 4], 'dtype': 'float32', 'lod_level': 1},
            'gt_class': {'shape': [None, 1], 'dtype': 'int32',   'lod_level': 1},
            'gt_score': {'shape': [None, 1], 'dtype': 'float32', 'lod_level': 1},
            'is_crowd': {'shape': [None, 1], 'dtype': 'int32',   'lod_level': 1},
            'is_difficult': {'shape': [None, 1], 'dtype': 'int32', 'lod_level': 1}
        }
        # yapf: disable
        if 'solo_target' in fields:
            n_features = len(self.solo_head.strides)
            targets_def = {}
            C = self.solo_head.cate_out_channels
            for lid in range(n_features):
                targets_def['layer%d_gt_objs' % lid] =    {'shape': [None, None, None, 1],    'dtype': 'float32', 'lod_level': 0}
                targets_def['layer%d_gt_clss' % lid] =    {'shape': [None, None, None, C],    'dtype': 'float32', 'lod_level': 0}
                targets_def['layer%d_gt_masks' % lid] =   {'shape': [None, None, None, None], 'dtype': 'float32', 'lod_level': 0}
                targets_def['layer%d_gt_pos_idx' % lid] = {'shape': [None, None, 3],          'dtype': 'int32',   'lod_level': 0}
            # yapf: enable
            inputs_def.update(targets_def)
        return inputs_def

    def build_inputs(
            self,
            image_shape=[3, None, None],
            fields=['image', 'ori_shape', 'resize_shape', 'solo_target'],  # for-train
            use_dataloader=True,
            iterable=False):
        inputs_def = self._inputs_def(image_shape, fields)
        if "solo_target" in fields:
            for i in range(len(self.solo_head.strides)):
                fields.extend(
                    ['layer%d_gt_objs' % i, 'layer%d_gt_clss' % i, 'layer%d_gt_masks' % i, 'layer%d_gt_pos_idx' % i])
            fields.remove('solo_target')
        feed_vars = OrderedDict([(key, fluid.data(
            name=key,
            shape=inputs_def[key]['shape'],
            dtype=inputs_def[key]['dtype'],
            lod_level=inputs_def[key]['lod_level'])) for key in fields])
        loader = fluid.io.DataLoader.from_generator(
            feed_list=list(feed_vars.values()),
            capacity=16,
            use_double_buffer=True,
            iterable=iterable) if use_dataloader else None
        return feed_vars, loader

    def train(self, feed_vars):
        return self.build(feed_vars, 'train')

    def eval(self, feed_vars):
        return self.build(feed_vars, 'test')

    def test(self, feed_vars, exclude_nms=False):
        assert not exclude_nms, "exclude_nms for {} is not support currently".format(
            self.__class__.__name__)
        return self.build(feed_vars, 'test')
