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
from __future__ import unicode_literals

import numpy as np
from PIL import Image, ImageDraw

from .colormap import colormap

__all__ = ['visualize_results']


def visualize_results(image,
                      im_id,
                      catid2name,
                      threshold=0.5,
                      bbox_results=None,
                      mask_results=None,
                      lmk_results=None):
    """
    Visualize bbox and mask results
    """
    if mask_results:
        image = draw_mask(image, im_id, mask_results, threshold)
    if bbox_results:
        image = draw_bbox(image, im_id, catid2name, bbox_results, threshold)
    if lmk_results:
        image = draw_lmk(image, im_id, lmk_results, threshold)
    return image


def draw_mask(image, im_id, segms, threshold, alpha=0.7):
    """
    Draw mask on image
    """
    mask_color_id = 0
    w_ratio = .4
    color_list = colormap(rgb=True)
    img_array = np.array(image).astype('float32')
    for dt in np.array(segms):
        if im_id != dt['image_id']:
            continue
        segm, score = dt['segmentation'], dt['score']
        if score < threshold:
            continue
        import pycocotools.mask as mask_util
        mask = mask_util.decode(segm) * 255
        color_mask = color_list[mask_color_id % len(color_list), 0:3]
        mask_color_id += 1
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        img_array[idx[0], idx[1], :] *= 1.0 - alpha
        img_array[idx[0], idx[1], :] += alpha * color_mask
    return Image.fromarray(img_array.astype('uint8'))


def draw_bbox(image, im_id, catid2name, bboxes, threshold):
    """
    Draw bbox on image
    """
    draw = ImageDraw.Draw(image)

    catid2color = {}
    color_list = colormap(rgb=True)[:40]
    for dt in np.array(bboxes):
        if im_id != dt['image_id']:
            continue
        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold:
            continue

        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h

        if catid not in catid2color:
            idx = np.random.randint(len(color_list))
            catid2color[catid] = color_list[idx]
        color = tuple(catid2color[catid])

        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=2,
            fill=color)

        # draw label
        text = "{} {:.2f}".format(catid2name[catid], score)
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

    return image


def draw_lmk(image, im_id, lmk_results, threshold):
    draw = ImageDraw.Draw(image)
    catid2color = {}
    color_list = colormap(rgb=True)[:40]
    for dt in np.array(lmk_results):
        lmk_decode, score = dt['landmark'], dt['score']
        if im_id != dt['image_id']:
            continue
        if score < threshold:
            continue
        for j in range(5):
            x1 = int(round(lmk_decode[2 * j]))
            y1 = int(round(lmk_decode[2 * j + 1]))
            draw.ellipse(
                (x1, y1, x1 + 5, y1 + 5), fill='green', outline='green')
    return image


import cv2
import colorsys
import random

def get_colors(n_colors):
    hsv_tuples = [(1.0 * x / n_colors, 1., 1.) for x in range(n_colors)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    return colors

def draw(image, boxes, scores, classes, masks, clsid2catid, catid2name, colors, mask_alpha=0.45):
    image_h, image_w, _ = image.shape

    for box, score, cl, ms in zip(boxes, scores, classes, masks):
        # 框坐标
        x0, y0, x1, y1 = box
        left = max(0, np.floor(x0 + 0.5).astype(int))
        top = max(0, np.floor(y0 + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))

        # 随机颜色
        bbox_color = random.choice(colors)
        # 同一类别固定颜色
        # bbox_color = colors[cl * 7]

        # 在这里上掩码颜色。咩咩深度优化的画掩码代码。
        color = np.array(bbox_color)
        color = np.reshape(color, (1, 1, 3))
        target_ms = ms[top:bottom, left:right]
        target_ms = np.expand_dims(target_ms, axis=2)
        target_ms = np.tile(target_ms, (1, 1, 3))
        target_region = image[top:bottom, left:right, :]
        target_region = target_ms * (target_region * (1 - mask_alpha) + color * mask_alpha) + (1 - target_ms) * target_region
        image[top:bottom, left:right, :] = target_region


        # 画框
        bbox_thick = 1
        cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
        bbox_mess = '%s: %.2f' % (catid2name[clsid2catid[cl]], score)
        t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
        cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
        cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return image




