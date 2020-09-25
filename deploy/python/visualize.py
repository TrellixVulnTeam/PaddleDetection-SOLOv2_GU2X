# coding: utf-8
# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import division

import cv2
import numpy as np
from PIL import Image, ImageDraw


def visualize_box_mask(im, results, colors, labels):
    if isinstance(im, str):
        with open(im, 'rb') as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype='uint8')
        image = cv2.imdecode(data, 1)  # BGR mode
    else:
        image = im

    if 'boxes' in results:
        if len(results['boxes']) > 0:
            draw(image, results['boxes'], results['scores'], results['classes'], results['masks'], labels, colors)
    return image


def get_color_map_list(num_classes):
    """ 
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list 
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def expand_boxes(boxes, scale=0.0):
    """ 
    Args:
        boxes (np.ndarray): shape:[N,4], N:number of box，
                            matix element:[x_min, y_min, x_max, y_max]
        scale (float): scale of boxes
    Returns:
        boxes_exp (np.ndarray): expanded boxes
    """
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5
    w_half *= scale
    h_half *= scale
    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def draw_mask(im, np_boxes, np_masks, labels, resolution=14, threshold=0.5):
    """ 
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box，
                               matix element:[class, score, x_min, y_min, x_max, y_max]
        np_masks (np.ndarray): shape:[N, class_num, resolution, resolution]
        labels (list): labels:['class1', ..., 'classn']
        resolution (int): shape of a mask is:[resolution, resolution]
        threshold (float): threshold of mask
    Returns:
        im (PIL.Image.Image): visualized image  
    """
    color_list = get_color_map_list(len(labels))
    scale = (resolution + 2.0) / resolution
    im_w, im_h = im.size
    w_ratio = 0.4
    alpha = 0.7
    im = np.array(im).astype('float32')
    rects = np_boxes[:, 2:]
    expand_rects = expand_boxes(rects, scale)
    expand_rects = expand_rects.astype(np.int32)
    clsid_scores = np_boxes[:, 0:2]
    padded_mask = np.zeros((resolution + 2, resolution + 2), dtype=np.float32)
    clsid2color = {}
    for idx in range(len(np_boxes)):
        clsid, score = clsid_scores[idx].tolist()
        clsid = int(clsid)
        xmin, ymin, xmax, ymax = expand_rects[idx].tolist()
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)
        padded_mask[1:-1, 1:-1] = np_masks[idx, int(clsid), :, :]
        resized_mask = cv2.resize(padded_mask, (w, h))
        resized_mask = np.array(resized_mask > threshold, dtype=np.uint8)
        x0 = min(max(xmin, 0), im_w)
        x1 = min(max(xmax + 1, 0), im_w)
        y0 = min(max(ymin, 0), im_h)
        y1 = min(max(ymax + 1, 0), im_h)
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        im_mask[y0:y1, x0:x1] = resized_mask[(y0 - ymin):(y1 - ymin), (
            x0 - xmin):(x1 - xmin)]
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color_mask = clsid2color[clsid]
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(im_mask)
        color_mask = np.array(color_mask)
        im[idx[0], idx[1], :] *= 1.0 - alpha
        im[idx[0], idx[1], :] += alpha * color_mask
    return Image.fromarray(im.astype('uint8'))


def draw_box(im, np_boxes, labels):
    """ 
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box，
                               matix element:[class, score, x_min, y_min, x_max, y_max]
        labels (list): labels:['class1', ..., 'classn']
    Returns:
        im (PIL.Image.Image): visualized image  
    """
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    clsid2color = {}
    color_list = get_color_map_list(len(labels))

    for dt in np_boxes:
        clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color = tuple(clsid2color[clsid])

        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=draw_thickness,
            fill=color)

        # draw label
        text = "{} {:.4f}".format(labels[clsid], score)
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
    return im



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




