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

import logging
import numpy as np
import os
import time

import paddle.fluid as fluid

from .voc_eval import bbox_eval as voc_bbox_eval
from .post_process import mstest_box_post_process, mstest_mask_post_process, box_flip

__all__ = ['parse_fetches', 'eval_run', 'eval_results']

logger = logging.getLogger(__name__)


def parse_fetches(fetches, prog=None, extra_keys=None):
    """
    Parse fetch variable infos from model fetches,
    values for fetch_list and keys for stat
    """
    keys, values = [], []
    cls = []
    for k, v in fetches.items():
        if hasattr(v, 'name'):
            keys.append(k)
            #v.persistable = True
            values.append(v.name)
        else:
            cls.append(v)

    if prog is not None and extra_keys is not None:
        for k in extra_keys:
            try:
                v = fluid.framework._get_var(k, prog)
                keys.append(k)
                values.append(v.name)
            except Exception:
                pass

    return keys, values, cls


def length2lod(length_lod):
    offset_lod = [0]
    for i in length_lod:
        offset_lod.append(offset_lod[-1] + i)
    return [offset_lod]


def get_sub_feed(input, place):
    new_dict = {}
    res_feed = {}
    key_name = ['bbox', 'im_info', 'im_id', 'im_shape', 'bbox_flip']
    for k in key_name:
        if k in input.keys():
            new_dict[k] = input[k]
    for k in input.keys():
        if 'image' in k:
            new_dict[k] = input[k]
    for k, v in new_dict.items():
        data_t = fluid.LoDTensor()
        data_t.set(v[0], place)
        if 'bbox' in k:
            lod = length2lod(v[1][0])
            data_t.set_lod(lod)
        res_feed[k] = data_t
    return res_feed


def clean_res(result, keep_name_list):
    clean_result = {}
    for k in result.keys():
        if k in keep_name_list:
            clean_result[k] = result[k]
    result.clear()
    return clean_result


def eval_run(exe,
             compile_program,
             loader,
             clsid2catid,
             catid2name,
             keys,
             values,
             cls,
             cfg=None,
             sub_prog=None,
             sub_keys=None,
             sub_values=None,
             resolution=None):
    """
    Run evaluation program, return program outputs.
    """
    import json
    import pycocotools.mask as maskUtils
    import shutil
    if os.path.exists('eval_results/bbox/'): shutil.rmtree('eval_results/bbox/')
    if os.path.exists('eval_results/mask/'): shutil.rmtree('eval_results/mask/')
    if not os.path.exists('eval_results/'): os.mkdir('eval_results/')
    os.mkdir('eval_results/bbox/')
    os.mkdir('eval_results/mask/')

    iter_id = 0
    if len(cls) != 0:
        values = []
        for i in range(len(cls)):
            _, accum_map = cls[i].get_map_var()
            cls[i].reset(exe)
            values.append(accum_map)

    images_num = 0
    start_time = time.time()
    has_bbox = 'bbox' in keys

    try:
        loader.start()
        while True:
            outs = exe.run(compile_program,
                           fetch_list=values,
                           return_numpy=False)
            res = {
                k: (np.array(v), v.recursive_sequence_lengths())
                for k, v in zip(keys, outs)
            }
            multi_scale_test = getattr(cfg, 'MultiScaleTEST', None)
            mask_multi_scale_test = multi_scale_test and 'Mask' in cfg.architecture

            if multi_scale_test:
                post_res = mstest_box_post_process(res, multi_scale_test,
                                                   cfg.num_classes)
                res.update(post_res)
            if mask_multi_scale_test:
                place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
                sub_feed = get_sub_feed(res, place)
                sub_prog_outs = exe.run(sub_prog,
                                        feed=sub_feed,
                                        fetch_list=sub_values,
                                        return_numpy=False)
                sub_prog_res = {
                    k: (np.array(v), v.recursive_sequence_lengths())
                    for k, v in zip(sub_keys, sub_prog_outs)
                }
                post_res = mstest_mask_post_process(sub_prog_res, cfg)
                res.update(post_res)
            if multi_scale_test:
                res = clean_res(
                    res, ['im_info', 'bbox', 'im_id', 'im_shape', 'mask'])
            if iter_id % 100 == 0:
                logger.info('Test iter {}'.format(iter_id))



            im_id = int(res['im_id'][0])
            im_name = '%.12d.jpg' % im_id
            masks = res['masks'][0]
            classes = res['classes'][0]
            scores = res['scores'][0]
            if scores[0] < 0.0:
                masks = np.array([])
                classes = np.array([])
                scores = np.array([])
                boxes = np.array([])
            else:
                # 获取boxes
                boxes = []
                for ms in masks:
                    sum_1 = np.sum(ms, axis=0)
                    x = np.where(sum_1 > 0.5)[0]
                    sum_2 = np.sum(ms, axis=1)
                    y = np.where(sum_2 > 0.5)[0]
                    if len(x) == 0:  # 掩码全是0的话（即没有一个像素是前景）
                        x0, x1, y0, y1 = 0, 1, 0, 1
                    else:
                        x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
                    boxes.append([x0, y0, x1, y1])
                boxes = np.array(boxes).astype(np.float32)

            n = len(boxes)
            bbox_data = []
            mask_data = []
            for p in range(n):
                # 1.bbox
                clsid = classes[p]
                score = scores[p]
                xmin, ymin, xmax, ymax = boxes[p]
                catid = (clsid2catid[int(clsid)])
                w = xmax - xmin + 1
                h = ymax - ymin + 1

                bbox = [xmin, ymin, w, h]
                # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
                bbox = [round(float(x) * 10) / 10 for x in bbox]
                bbox_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'bbox': bbox,
                    'score': float(score)
                }
                bbox_data.append(bbox_res)

                # 2.mask
                # segm = pycocotools.mask.encode(np.asfortranarray(masks[p].astype(np.uint8)))
                # segm['counts'] = segm['counts'].decode('ascii')  # json.dump doesn't like bytes strings
                segm = maskUtils.encode(np.asfortranarray(masks[p].astype(np.uint8)))
                segm['counts'] = segm['counts'].decode('utf8')

                mask_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'segmentation': segm,
                    'score': float(score)
                }
                mask_data.append(mask_res)
            path = 'eval_results/bbox/%s.json' % im_name.split('.')[0]
            with open(path, 'w') as f:
                json.dump(bbox_data, f)
            path = 'eval_results/mask/%s.json' % im_name.split('.')[0]
            with open(path, 'w') as f:
                json.dump(mask_data, f)
            iter_id += 1
            images_num += len(res['bbox'][1][0]) if has_bbox else 1
    except (StopIteration, fluid.core.EOFException):
        loader.reset()
    logger.info('Test finish iter {}'.format(iter_id))

    end_time = time.time()
    fps = images_num / (end_time - start_time)
    if has_bbox:
        logger.info('Total number of images: {}, inference time: {} fps.'.
                    format(images_num, fps))
    else:
        logger.info('Total iteration: {}, inference time: {} batch/s.'.format(
            images_num, fps))


def eval_results(metric,
                 dataset=None):
    """Evaluation for evaluation program results"""
    box_ap_stats = []
    if metric == 'COCO':
        from ppdet.utils.coco_eval import proposal_eval, bbox_eval, mask_eval
        anno_file = dataset.get_anno()
        box_ap_stats = bbox_eval(anno_file)
        mask_ap_stats = mask_eval(anno_file)
    else:
        pass
    return box_ap_stats, mask_ap_stats




