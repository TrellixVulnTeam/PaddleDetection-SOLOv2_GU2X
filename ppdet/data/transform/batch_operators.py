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

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import logging
import cv2
import numpy as np

from .operators import register_op, BaseOperator
from .op_helper import jaccard_overlap, gaussian2D

logger = logging.getLogger(__name__)

__all__ = [
    'PadBatch',
    'RandomShape',
    'PadMultiScaleTest',
    'Gt2YoloTarget',
    'Gt2FCOSTarget',
    'Gt2SOLOTarget',
    'Gt2TTFTarget',
]


@register_op
class PadBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0, use_padded_im_info=True):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.use_padded_im_info = use_padded_im_info

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples
        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)

        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        padding_batch = []
        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if 'gt_mask' in data:
                gt_mask = data['gt_mask']
                ms_h, ms_w, ms_c = gt_mask.shape[:]
                padding_ms = np.zeros(
                    (max_shape[1], max_shape[2], ms_c), dtype=np.float32)
                padding_ms[:ms_h, :ms_w, :] = gt_mask  # gt_mask贴在padding_ms的左上部分实现对齐
                padding_ms = padding_ms.transpose(2, 0, 1)   # [物体数, pad_h, pad_w]
                data['gt_mask'] = padding_ms
            if self.use_padded_im_info:
                data['im_info'][:2] = max_shape[1:3]
            if 'semantic' in data.keys() and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem

        return samples


@register_op
class RandomShape(BaseOperator):
    """
    Randomly reshape a batch. If random_inter is True, also randomly
    select one an interpolation algorithm [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
    cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]. If random_inter is
    False, use cv2.INTER_NEAREST.
    Args:
        sizes (list): list of int, random choose a size from these
        random_inter (bool): whether to randomly interpolation, defalut true.
    """

    def __init__(self, sizes=[], random_inter=False, resize_box=False):
        super(RandomShape, self).__init__()
        self.sizes = sizes
        self.random_inter = random_inter
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ] if random_inter else []
        self.resize_box = resize_box

    def __call__(self, samples, context=None):
        shape = np.random.choice(self.sizes)
        method = np.random.choice(self.interps) if self.random_inter \
            else cv2.INTER_NEAREST
        for i in range(len(samples)):
            im = samples[i]['image']
            h, w = im.shape[:2]
            scale_x = float(shape) / w
            scale_y = float(shape) / h
            im = cv2.resize(
                im, None, None, fx=scale_x, fy=scale_y, interpolation=method)
            samples[i]['image'] = im
            if self.resize_box and 'gt_bbox' in samples[i] and len(samples[0][
                    'gt_bbox']) > 0:
                scale_array = np.array([scale_x, scale_y] * 2, dtype=np.float32)
                samples[i]['gt_bbox'] = np.clip(samples[i]['gt_bbox'] *
                                                scale_array, 0,
                                                float(shape) - 1)
        return samples


@register_op
class PadMultiScaleTest(BaseOperator):
    """
    Pad the image so they can be divisible by a stride for multi-scale testing.
 
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0):
        super(PadMultiScaleTest, self).__init__()
        self.pad_to_stride = pad_to_stride

    def __call__(self, samples, context=None):
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples

        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        if len(samples) != 1:
            raise ValueError("Batch size must be 1 when using multiscale test, "
                             "but now batch size is {}".format(len(samples)))
        for i in range(len(samples)):
            sample = samples[i]
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im_c, im_h, im_w = im.shape
                    max_h = int(
                        np.ceil(im_h / coarsest_stride) * coarsest_stride)
                    max_w = int(
                        np.ceil(im_w / coarsest_stride) * coarsest_stride)
                    padding_im = np.zeros(
                        (im_c, max_h, max_w), dtype=np.float32)

                    padding_im[:, :im_h, :im_w] = im
                    sample[k] = padding_im
                    info_name = 'im_info' if k == 'image' else 'im_info_' + k
                    # update im_info
                    sample[info_name][:2] = [max_h, max_w]
        if not batch_input:
            samples = samples[0]
        return samples


@register_op
class Gt2YoloTarget(BaseOperator):
    """
    Generate YOLOv3 targets by groud truth data, this operator is only used in
    fine grained YOLOv3 loss mode
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.):
        super(Gt2YoloTarget, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = samples[0]['image'].shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            for i, (
                    mask, downsample_ratio
            ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                for b in range(gt_bbox.shape[0]):
                    gx, gy, gw, gh = gt_bbox[b, :]
                    cls = gt_class[b]
                    score = gt_score[b]
                    if gw <= 0. or gh <= 0. or score <= 0.:
                        continue

                    # find best match anchor index
                    best_iou = 0.
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx

                    gi = int(gx * grid_w)
                    gj = int(gy * grid_h)

                    # gtbox should be regresed in this layes if best match 
                    # anchor index in anchor mask of this layer
                    if best_idx in mask:
                        best_n = mask.index(best_idx)

                        # x, y, w, h, scale
                        target[best_n, 0, gj, gi] = gx * grid_w - gi
                        target[best_n, 1, gj, gi] = gy * grid_h - gj
                        target[best_n, 2, gj, gi] = np.log(
                            gw * w / self.anchors[best_idx][0])
                        target[best_n, 3, gj, gi] = np.log(
                            gh * h / self.anchors[best_idx][1])
                        target[best_n, 4, gj, gi] = 2.0 - gw * gh

                        # objectness record gt_score
                        target[best_n, 5, gj, gi] = score

                        # classification
                        target[best_n, 6 + cls, gj, gi] = 1.

                    # For non-matched anchors, calculate the target if the iou 
                    # between anchor and gt is larger than iou_thresh
                    if self.iou_thresh < 1:
                        for idx, mask_i in enumerate(mask):
                            if mask_i == best_idx: continue
                            iou = jaccard_overlap(
                                [0., 0., gw, gh],
                                [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                            if iou > self.iou_thresh:
                                # x, y, w, h, scale
                                target[idx, 0, gj, gi] = gx * grid_w - gi
                                target[idx, 1, gj, gi] = gy * grid_h - gj
                                target[idx, 2, gj, gi] = np.log(
                                    gw * w / self.anchors[mask_i][0])
                                target[idx, 3, gj, gi] = np.log(
                                    gh * h / self.anchors[mask_i][1])
                                target[idx, 4, gj, gi] = 2.0 - gw * gh

                                # objectness record gt_score
                                target[idx, 5, gj, gi] = score

                                # classification
                                target[idx, 6 + cls, gj, gi] = 1.
                sample['target{}'.format(i)] = target
        return samples


@register_op
class Gt2FCOSTarget(BaseOperator):
    """
    Generate FCOS targets by groud truth data
    """

    def __init__(self,
                 object_sizes_boundary,
                 center_sampling_radius,
                 downsample_ratios,
                 norm_reg_targets=False):
        super(Gt2FCOSTarget, self).__init__()
        self.center_sampling_radius = center_sampling_radius
        self.downsample_ratios = downsample_ratios
        self.INF = np.inf
        self.object_sizes_boundary = [-1] + object_sizes_boundary + [self.INF]
        object_sizes_of_interest = []
        for i in range(len(self.object_sizes_boundary) - 1):
            object_sizes_of_interest.append([
                self.object_sizes_boundary[i], self.object_sizes_boundary[i + 1]
            ])
        self.object_sizes_of_interest = object_sizes_of_interest
        self.norm_reg_targets = norm_reg_targets

    def _compute_points(self, w, h):
        """
        compute the corresponding points in each feature map
        :param h: image height
        :param w: image width
        :return: points from all feature map
        """
        locations = []
        for stride in self.downsample_ratios:
            shift_x = np.arange(0, w, stride).astype(np.float32)
            shift_y = np.arange(0, h, stride).astype(np.float32)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()
            location = np.stack([shift_x, shift_y], axis=1) + stride // 2
            locations.append(location)
        num_points_each_level = [len(location) for location in locations]
        locations = np.concatenate(locations, axis=0)
        return locations, num_points_each_level

    def _convert_xywh2xyxy(self, gt_bbox, w, h):
        """
        convert the bounding box from style xywh to xyxy
        :param gt_bbox: bounding boxes normalized into [0, 1]
        :param w: image width
        :param h: image height
        :return: bounding boxes in xyxy style
        """
        bboxes = gt_bbox.copy()
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        return bboxes

    def _check_inside_boxes_limited(self, gt_bbox, xs, ys,
                                    num_points_each_level):
        """
        check if points is within the clipped boxes
        :param gt_bbox: bounding boxes
        :param xs: horizontal coordinate of points
        :param ys: vertical coordinate of points
        :return: the mask of points is within gt_box or not
        """
        bboxes = np.reshape(
            gt_bbox, newshape=[1, gt_bbox.shape[0], gt_bbox.shape[1]])
        bboxes = np.tile(bboxes, reps=[xs.shape[0], 1, 1])
        ct_x = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2
        ct_y = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2
        beg = 0
        clipped_box = bboxes.copy()
        for lvl, stride in enumerate(self.downsample_ratios):
            end = beg + num_points_each_level[lvl]
            stride_exp = self.center_sampling_radius * stride
            clipped_box[beg:end, :, 0] = np.maximum(
                bboxes[beg:end, :, 0], ct_x[beg:end, :] - stride_exp)
            clipped_box[beg:end, :, 1] = np.maximum(
                bboxes[beg:end, :, 1], ct_y[beg:end, :] - stride_exp)
            clipped_box[beg:end, :, 2] = np.minimum(
                bboxes[beg:end, :, 2], ct_x[beg:end, :] + stride_exp)
            clipped_box[beg:end, :, 3] = np.minimum(
                bboxes[beg:end, :, 3], ct_y[beg:end, :] + stride_exp)
            beg = end
        l_res = xs - clipped_box[:, :, 0]
        r_res = clipped_box[:, :, 2] - xs
        t_res = ys - clipped_box[:, :, 1]
        b_res = clipped_box[:, :, 3] - ys
        clipped_box_reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)
        inside_gt_box = np.min(clipped_box_reg_targets, axis=2) > 0
        return inside_gt_box

    def __call__(self, samples, context=None):
        assert len(self.object_sizes_of_interest) == len(self.downsample_ratios), \
            "object_sizes_of_interest', and 'downsample_ratios' should have same length."

        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            im_info = sample['im_info']
            bboxes = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * np.floor(im_info[1]) / \
                np.floor(im_info[1] / im_info[2])
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * np.floor(im_info[0]) / \
                np.floor(im_info[0] / im_info[2])
            # calculate the locations
            h, w = sample['image'].shape[1:3]
            points, num_points_each_level = self._compute_points(w, h)
            object_scale_exp = []
            for i, num_pts in enumerate(num_points_each_level):
                object_scale_exp.append(
                    np.tile(
                        np.array([self.object_sizes_of_interest[i]]),
                        reps=[num_pts, 1]))
            object_scale_exp = np.concatenate(object_scale_exp, axis=0)

            gt_area = (bboxes[:, 2] - bboxes[:, 0]) * (
                bboxes[:, 3] - bboxes[:, 1])
            xs, ys = points[:, 0], points[:, 1]
            xs = np.reshape(xs, newshape=[xs.shape[0], 1])
            xs = np.tile(xs, reps=[1, bboxes.shape[0]])
            ys = np.reshape(ys, newshape=[ys.shape[0], 1])
            ys = np.tile(ys, reps=[1, bboxes.shape[0]])

            l_res = xs - bboxes[:, 0]
            r_res = bboxes[:, 2] - xs
            t_res = ys - bboxes[:, 1]
            b_res = bboxes[:, 3] - ys
            reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)
            if self.center_sampling_radius > 0:
                is_inside_box = self._check_inside_boxes_limited(
                    bboxes, xs, ys, num_points_each_level)
            else:
                is_inside_box = np.min(reg_targets, axis=2) > 0
            # check if the targets is inside the corresponding level
            max_reg_targets = np.max(reg_targets, axis=2)
            lower_bound = np.tile(
                np.expand_dims(
                    object_scale_exp[:, 0], axis=1),
                reps=[1, max_reg_targets.shape[1]])
            high_bound = np.tile(
                np.expand_dims(
                    object_scale_exp[:, 1], axis=1),
                reps=[1, max_reg_targets.shape[1]])
            is_match_current_level = \
                (max_reg_targets > lower_bound) & \
                (max_reg_targets < high_bound)
            points2gtarea = np.tile(
                np.expand_dims(
                    gt_area, axis=0), reps=[xs.shape[0], 1])
            points2gtarea[is_inside_box == 0] = self.INF
            points2gtarea[is_match_current_level == 0] = self.INF
            points2min_area = points2gtarea.min(axis=1)
            points2min_area_ind = points2gtarea.argmin(axis=1)
            labels = gt_class[points2min_area_ind] + 1
            labels[points2min_area == self.INF] = 0
            reg_targets = reg_targets[range(xs.shape[0]), points2min_area_ind]
            ctn_targets = np.sqrt((reg_targets[:, [0, 2]].min(axis=1) / \
                                  reg_targets[:, [0, 2]].max(axis=1)) * \
                                  (reg_targets[:, [1, 3]].min(axis=1) / \
                                   reg_targets[:, [1, 3]].max(axis=1))).astype(np.float32)
            ctn_targets = np.reshape(
                ctn_targets, newshape=[ctn_targets.shape[0], 1])
            ctn_targets[labels <= 0] = 0
            pos_ind = np.nonzero(labels != 0)
            reg_targets_pos = reg_targets[pos_ind[0], :]
            split_sections = []
            beg = 0
            for lvl in range(len(num_points_each_level)):
                end = beg + num_points_each_level[lvl]
                split_sections.append(end)
                beg = end
            labels_by_level = np.split(labels, split_sections, axis=0)
            reg_targets_by_level = np.split(reg_targets, split_sections, axis=0)
            ctn_targets_by_level = np.split(ctn_targets, split_sections, axis=0)
            for lvl in range(len(self.downsample_ratios)):
                grid_w = int(np.ceil(w / self.downsample_ratios[lvl]))
                grid_h = int(np.ceil(h / self.downsample_ratios[lvl]))
                if self.norm_reg_targets:
                    sample['reg_target{}'.format(lvl)] = \
                        np.reshape(
                            reg_targets_by_level[lvl] / \
                            self.downsample_ratios[lvl],
                            newshape=[grid_h, grid_w, 4])
                else:
                    sample['reg_target{}'.format(lvl)] = np.reshape(
                        reg_targets_by_level[lvl],
                        newshape=[grid_h, grid_w, 4])
                sample['labels{}'.format(lvl)] = np.reshape(
                    labels_by_level[lvl], newshape=[grid_h, grid_w, 1])
                sample['centerness{}'.format(lvl)] = np.reshape(
                    ctn_targets_by_level[lvl], newshape=[grid_h, grid_w, 1])
        return samples


from scipy import ndimage

@register_op
class Gt2SOLOTarget(BaseOperator):
    def __init__(self,
                 num_classes=80,
                 strides=[8, 8, 16, 32, 32],
                 scale_ranges=[[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]],
                 sigma=0.2,
                 num_grids=[40, 36, 24, 16, 12]):
        super(Gt2SOLOTarget, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.strides = strides
        self.scale_ranges = scale_ranges
        self.sigma = sigma
        self.max_pos = 600

    def __call__(self, samples, context=None):
        h, w = samples[0]['image'].shape[1:]
        featmap_sizes = []
        for st in self.strides:
            featmap_sizes.append([h*2//st, w*2//st])

        batch_size = len(samples)
        n_features = len(self.seg_num_grids)
        max_masks_num = [-1] * n_features   # 每个输出层最多的gt掩码个数，用于对齐gt掩码
        for i in range(batch_size):
            gt_bbox = samples[i]['gt_bbox']
            gt_score = samples[i]['gt_score']
            gt_class = samples[i]['gt_class']
            gt_mask = samples[i]['gt_mask']
            # gt_mask = gt_mask.transpose(2, 0, 1)
            # 去掉添加的负分数物体。
            gt_num = 0
            for q in range(len(gt_score)):
                if gt_score[q] > 0:
                    gt_num += 1
                else:
                    break
            gt_bbox = gt_bbox[:gt_num]
            gt_class = gt_class[:gt_num]
            gt_mask = gt_mask[:gt_num]
            gt_objs_per_layer, gt_clss_per_layer, gt_masks_per_layer, gt_pos_idx_per_layer = self.solo_target_single(
                gt_bbox, gt_class, gt_mask, featmap_sizes)
            for lvl in range(n_features):
                samples[i]['layer%d_gt_objs' % lvl] = gt_objs_per_layer[lvl]
                samples[i]['layer%d_gt_clss' % lvl] = gt_clss_per_layer[lvl]
                samples[i]['layer%d_gt_masks' % lvl] = gt_masks_per_layer[lvl]
                samples[i]['layer%d_gt_pos_idx' % lvl] = gt_pos_idx_per_layer[lvl]
                masks_num = len(gt_masks_per_layer[lvl])
                if masks_num > max_masks_num[lvl]:
                    max_masks_num[lvl] = masks_num
        # 对齐gt掩码
        for i in range(batch_size):
            for lvl in range(n_features):
                gt_masks = samples[i]['layer%d_gt_masks' % lvl]
                padding_gt_masks = np.zeros((max_masks_num[lvl], gt_masks.shape[1], gt_masks.shape[2]), dtype=np.float32)
                padding_gt_masks[:gt_masks.shape[0], :, :] = gt_masks
                samples[i]['layer%d_gt_masks' % lvl] = padding_gt_masks
                gt_pos_idx = samples[i]['layer%d_gt_pos_idx' % lvl]
                samples[i]['layer%d_gt_pos_idx' % lvl] = gt_pos_idx[:max_masks_num[lvl], :]
        return samples

    def solo_target_single(self,
                           gt_bboxes_raw,
                           gt_labels_raw,
                           gt_masks_raw,
                           featmap_sizes=None):
        # ins
        gt_areas = np.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (   # 平均边长，几何平均数， [n, ]
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        gt_objs_per_layer = []
        gt_clss_per_layer = []
        gt_masks_per_layer = []
        gt_pos_idx_per_layer = []
        # 遍历每个输出层
        #           (1,     96)            8       40
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.seg_num_grids):
            # [40, 40, 1]  objectness
            gt_objs = np.zeros([num_grid, num_grid, 1], dtype=np.float32)
            # [40, 40, 80]  种类one-hot
            gt_clss = np.zeros([num_grid, num_grid, self.num_classes], dtype=np.float32)
            # [?, 104, 104]  这一输出层的gt_masks，可能同一个掩码重复多次
            gt_masks = []
            # [self.max_pos, 3]    坐标以-2初始化
            # 前2个用于把正样本抽出来gather_nd()，后1个用于把掩码抽出来gather()。为了避免使用layers.where()后顺序没对上，所以不拆开写。
            gt_pos_idx = np.zeros([self.max_pos, 3], dtype=np.int32) - 2
            # 掩码计数
            p = 0

            # 这一张图片，所有物体，若平均边长在这个范围，这一输出层就负责预测。因为面积范围有交集，所以一个gt可以被分配到多个输出层上。
            hit_indices = np.where((gt_areas >= lower_bound) & (gt_areas <= upper_bound))[0]

            if len(hit_indices) == 0:   # 这一层没有正样本
                gt_objs_per_layer.append(gt_objs)   # 全是0
                gt_clss_per_layer.append(gt_clss)   # 全是0
                gt_masks = np.zeros([1, featmap_sizes[0][0], featmap_sizes[0][1]], dtype=np.uint8)   # 全是0，至少一张掩码，方便gather()
                gt_masks_per_layer.append(gt_masks)
                gt_pos_idx[0, :] = np.array([0, 0, 0], dtype=np.int32)   # 没有正样本，默认会抽第0行第0列格子，默认会抽这一层gt_mask里第0个掩码。
                gt_pos_idx_per_layer.append(gt_pos_idx)
                continue
            gt_bboxes_raw_this_layer = gt_bboxes_raw[hit_indices]   # shape=[m, 4]  这一层负责预测的物体的bbox
            gt_labels_raw_this_layer = gt_labels_raw[hit_indices]   # shape=[m, ]   这一层负责预测的物体的类别id
            gt_masks_raw_this_layer = gt_masks_raw[hit_indices]   # [m, ?, ?]

            half_ws = 0.5 * (gt_bboxes_raw_this_layer[:, 2] - gt_bboxes_raw_this_layer[:, 0]) * self.sigma   # shape=[m, ]  宽的一半
            half_hs = 0.5 * (gt_bboxes_raw_this_layer[:, 3] - gt_bboxes_raw_this_layer[:, 1]) * self.sigma   # shape=[m, ]  高的一半

            output_stride = 4   # 因为网络生成的 掩码原型宽高 是 输入图片宽高 的1/4。 所以将真实掩码下采样4倍。

            for seg_mask, gt_label, half_h, half_w in zip(gt_masks_raw_this_layer, gt_labels_raw_this_layer, half_hs, half_ws):
                if seg_mask.sum() < 1:   # 忽略太小的物体
                   continue
                # mass center
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)   # 也就是输入图片的大小
                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)    # 求物体掩码的质心。scipy提供技术支持。
                # seg_mask.sum()是0时，center_w是数值nan
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))      # 物体质心落在了第几列格子
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))      # 物体质心落在了第几行格子

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))      # 物体左上角落在了第几行格子
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))    # 物体右下角落在了第几行格子
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))     # 物体左上角落在了第几列格子
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))   # 物体右下角落在了第几列格子

                # 物体的宽高并没有那么重要。将物体的左上角、右下角限制在质心所在的九宫格内。当物体很小时，物体的左上角、右下角、质心位于同一个格子。
                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)


                # 40x40的网格，将负责预测gt的格子填上gt_objs和gt_clss，此处同YOLOv3
                # ins  [img_h, img_w]->[img_h/output_stride, img_w/output_stride]  将gt的掩码下采样output_stride=4倍。
                seg_mask = cv2.resize(seg_mask, None, None, fx=1. / output_stride, fy=1. / output_stride, interpolation=cv2.INTER_LINEAR)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        if gt_objs[i, j, 0] < 0.5:   # 这个格子没有被填过gt才可以填。
                            gt_objs[i, j, 0] = 1.0   # 此处同YOLOv3
                            gt_clss[i, j, gt_label] = 1.0   # 此处同YOLOv3
                            cp_mask = np.copy(seg_mask)
                            cp_mask = cp_mask[np.newaxis, :, :]
                            gt_masks.append(cp_mask)
                            gt_pos_idx[p, :] = np.array([i, j, p], dtype=np.int32)   # 前2个用于把正样本抽出来gather_nd()，后1个用于把掩码抽出来gather()。
                            p += 1
            # 平均边长在这个范围内，但是因为面积太小seg_mask.sum() < 1导致continue，导致这一输出层也没有正样本的时候，也分配一个。
            if len(gt_masks) == 0:
                gt_masks = np.zeros([1, featmap_sizes[0][0], featmap_sizes[0][1]], dtype=np.uint8)   # 全是0，至少一张掩码，方便gather()
                gt_pos_idx[0, :] = np.array([0, 0, 0], dtype=np.int32)   # 没有正样本，默认会抽第0行第0列格子，默认会抽这一层gt_mask里第0个掩码。
            else:
                gt_masks = np.concatenate(gt_masks, axis=0)

            gt_objs_per_layer.append(gt_objs)
            gt_clss_per_layer.append(gt_clss)
            gt_masks_per_layer.append(gt_masks)
            gt_pos_idx_per_layer.append(gt_pos_idx)
        return gt_objs_per_layer, gt_clss_per_layer, gt_masks_per_layer, gt_pos_idx_per_layer



@register_op
class Gt2TTFTarget(BaseOperator):
    """
    Gt2TTFTarget
    Generate TTFNet targets by ground truth data
    
    Args:
        num_classes(int): the number of classes.
        down_ratio(int): the down ratio from images to heatmap, 4 by default.
        alpha(float): the alpha parameter to generate gaussian target.
            0.54 by default.
    """

    def __init__(self, num_classes, down_ratio=4, alpha=0.54):
        super(Gt2TTFTarget, self).__init__()
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.alpha = alpha

    def __call__(self, samples, context=None):
        output_size = samples[0]['image'].shape[1]
        feat_size = output_size // self.down_ratio
        for sample in samples:
            heatmap = np.zeros(
                (self.num_classes, feat_size, feat_size), dtype='float32')
            box_target = np.ones(
                (4, feat_size, feat_size), dtype='float32') * -1
            reg_weight = np.zeros((1, feat_size, feat_size), dtype='float32')

            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']

            bbox_w = gt_bbox[:, 2] - gt_bbox[:, 0] + 1
            bbox_h = gt_bbox[:, 3] - gt_bbox[:, 1] + 1
            area = bbox_w * bbox_h
            boxes_areas_log = np.log(area)
            boxes_ind = np.argsort(boxes_areas_log, axis=0)[::-1]
            boxes_area_topk_log = boxes_areas_log[boxes_ind]
            gt_bbox = gt_bbox[boxes_ind]
            gt_class = gt_class[boxes_ind]

            feat_gt_bbox = gt_bbox / self.down_ratio
            feat_gt_bbox = np.clip(feat_gt_bbox, 0, feat_size - 1)
            feat_hs, feat_ws = (feat_gt_bbox[:, 3] - feat_gt_bbox[:, 1],
                                feat_gt_bbox[:, 2] - feat_gt_bbox[:, 0])

            ct_inds = np.stack(
                [(gt_bbox[:, 0] + gt_bbox[:, 2]) / 2,
                 (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2],
                axis=1) / self.down_ratio

            h_radiuses_alpha = (feat_hs / 2. * self.alpha).astype('int32')
            w_radiuses_alpha = (feat_ws / 2. * self.alpha).astype('int32')

            for k in range(len(gt_bbox)):
                cls_id = gt_class[k]
                fake_heatmap = np.zeros((feat_size, feat_size), dtype='float32')
                self.draw_truncate_gaussian(fake_heatmap, ct_inds[k],
                                            h_radiuses_alpha[k],
                                            w_radiuses_alpha[k])

                heatmap[cls_id] = np.maximum(heatmap[cls_id], fake_heatmap)
                box_target_inds = fake_heatmap > 0
                box_target[:, box_target_inds] = gt_bbox[k][:, None]

                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = np.sum(local_heatmap)
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[0, box_target_inds] = local_heatmap / ct_div
            sample['ttf_heatmap'] = heatmap
            sample['ttf_box_target'] = box_target
            sample['ttf_reg_weight'] = reg_weight
        return samples

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = gaussian2D((h, w), sigma_x, sigma_y)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius -
                                   left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            heatmap[y - top:y + bottom, x - left:x + right] = np.maximum(
                masked_heatmap, masked_gaussian)
        return heatmap
