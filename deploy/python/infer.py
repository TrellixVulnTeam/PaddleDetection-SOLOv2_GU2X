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

import os
import argparse
import time
import yaml
import ast
from functools import reduce

from PIL import Image
import cv2
import numpy as np
import paddle.fluid as fluid


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

def draw(image, boxes, scores, classes, masks, clsid2name, colors, random_color=True, mask_alpha=0.45):
    image_h, image_w, _ = image.shape

    for box, score, cl, ms in zip(boxes, scores, classes, masks):
        # 框坐标
        x0, y0, x1, y1 = box
        left = max(0, np.floor(x0 + 0.5).astype(int))
        top = max(0, np.floor(y0 + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))

        # 随机颜色
        if random_color:
            bbox_color = random.choice(colors)
        # 同一类别固定颜色
        else:
            bbox_color = colors[cl * 7]

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
        bbox_mess = '%s: %.2f' % (clsid2name[cl], score)
        t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
        cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
        cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return image





def decode_image(im_file, im_info):
    """read rgb image
    Args:
        im_file (str/np.ndarray): path of image/ np.ndarray read by cv2
        im_info (dict): info of image
    Returns:
        im (np.ndarray):  processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    if isinstance(im_file, str):
        with open(im_file, 'rb') as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # im_info['origin_shape'] = im.shape[:2]
        # im_info['resize_shape'] = im.shape[:2]
        im_info['ori_shape'] = im.shape
        im_info['resize_shape'] = im.shape
    else:
        im = im_file

        # BGR mode, but need RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # im_info['origin_shape'] = im.shape[:2]
        # im_info['resize_shape'] = im.shape[:2]
        im_info['ori_shape'] = im.shape
        im_info['resize_shape'] = im.shape
    return im, im_info


class Resize(object):
    def __init__(self,
                 target_size=0,
                 max_size=0,
                 interp=cv2.INTER_LINEAR,
                 use_cv2=True):
        self.target_size = target_size
        self.max_size = int(max_size)
        self.interp = int(interp)
        self.use_cv2 = use_cv2

    def __call__(self, im, im_info):
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        selected_size = self.target_size
        if float(im_size_min) == 0:
            raise ZeroDivisionError('{}: min size of image is 0'.format(self))
        if self.max_size != 0:
            im_scale = float(selected_size) / float(im_size_min)    # 短的边 缩放到选中的尺度
            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > self.max_size:    # 如果缩放后 长的边 超过了self.max_size=1333，那就改为将 长的边 缩放到self.max_size
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale   # 不破坏原始宽高比地缩放
            im_scale_y = im_scale   # 不破坏原始宽高比地缩放

            resize_w = im_scale_x * float(im_shape[1])
            resize_h = im_scale_y * float(im_shape[0])
        else:
            im_scale_x = float(selected_size) / float(im_shape[1])
            im_scale_y = float(selected_size) / float(im_shape[0])

            resize_w = selected_size
            resize_h = selected_size

        if self.use_cv2:
            im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            im_info['resize_shape'] = im.shape
        else:
            if self.max_size != 0:
                raise TypeError(
                    'If you set max_size to cap the maximum size of image,'
                    'please set use_cv2 to True to resize the image.')
            im = im.astype('uint8')
            im = Image.fromarray(im)
            im = im.resize((int(resize_w), int(resize_h)), self.interp)
            im = np.array(im)
        return im, im_info


class Normalize(object):
    """normalize image
    Args:
        mean (list): im - mean
        std (list): im / std
        is_scale (bool): whether need im / 255
        is_channel_first (bool): if True: image shape is CHW, else: HWC
    """

    def __init__(self, mean, std, is_scale=True, is_channel_first=False):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.astype(np.float32, copy=False)
        if self.is_channel_first:
            mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
            std = np.array(self.std)[:, np.newaxis, np.newaxis]
        else:
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
        if self.is_scale:
            im = im / 255.0
        im -= mean
        im /= std
        return im, im_info


class Permute(object):
    """permute image
    Args:
        to_bgr (bool): whether convert RGB to BGR 
        channel_first (bool): whether convert HWC to CHW
    """

    def __init__(self, to_bgr=False, channel_first=True):
        self.to_bgr = to_bgr
        self.channel_first = channel_first

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        if self.channel_first:
            im = im.transpose((2, 0, 1)).copy()
        if self.to_bgr:
            im = im[[2, 1, 0], :, :]
        return im, im_info


class PadStride(object):
    def __init__(self, stride=0):
        self.pad_to_stride = stride

    def __call__(self, im, im_info):
        coarsest_stride = self.pad_to_stride
        max_shape = np.array(im.shape)    # max_shape=[3, max_h, max_w]

        max_shape[1] = int(   # max_h增加到最小的能被coarsest_stride=128整除的数
            np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
        max_shape[2] = int(   # max_w增加到最小的能被coarsest_stride=128整除的数
            np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        im_c, im_h, im_w = im.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im    # im贴在padding_im的左上部分实现对齐
        return padding_im, im_info


def create_inputs(im, im_info, model_arch='YOLO'):
    """generate input for different model type
    Args:
        im (np.ndarray): image (np.ndarray)
        im_info (dict): info of image
        model_arch (str): model type
    Returns:
        inputs (dict): input of model
    """
    inputs = {}
    inputs['image'] = im
    inputs['ori_shape'] = np.array([im_info['ori_shape']]).astype('int32')
    inputs['resize_shape'] = np.array([im_info['resize_shape']]).astype('int32')
    # origin_shape = list(im_info['origin_shape'])
    # resize_shape = list(im_info['resize_shape'])
    # scale_x, scale_y = im_info['scale']
    # if 'YOLO' in model_arch:
    #     im_size = np.array([origin_shape]).astype('int32')
    #     inputs['im_size'] = im_size
    # elif 'SOLOv2' in model_arch:
    #     resize_shape = np.array([resize_shape + [1.]]).astype('int32')
    #     origin_shape = np.array([origin_shape + [1.]]).astype('int32')
    #     inputs['resize_shape'] = resize_shape
    #     inputs['ori_shape'] = origin_shape
    return inputs


class Config():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """
    support_models = ['YOLO', 'SSD', 'RetinaNet', 'RCNN', 'Face', 'TTF', 'SOLOv2']

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.use_python_inference = yml_conf['use_python_inference']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask_resolution = None
        if 'mask_resolution' in yml_conf:
            self.mask_resolution = yml_conf['mask_resolution']
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type 
        """
        for support_model in self.support_models:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], self.support_models))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: %s' % ('Use Padddle Executor', self.use_python_inference))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def load_predictor(model_dir,
                   run_mode='fluid',
                   batch_size=1,
                   use_gpu=False,
                   min_subgraph_size=3):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        use_gpu (bool): whether use gpu
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need use_gpu == True.
    """
    if not use_gpu and not run_mode == 'fluid':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect use_gpu==True, but use_gpu == {}"
            .format(run_mode, use_gpu))
    if run_mode == 'trt_int8':
        raise ValueError("TensorRT int8 mode is not supported now, "
                         "please use trt_fp32 or trt_fp16 instead.")
    precision_map = {
        'trt_int8': fluid.core.AnalysisConfig.Precision.Int8,
        'trt_fp32': fluid.core.AnalysisConfig.Precision.Float32,
        'trt_fp16': fluid.core.AnalysisConfig.Precision.Half
    }
    config = fluid.core.AnalysisConfig(
        os.path.join(model_dir, '__model__'),
        os.path.join(model_dir, '__params__'))
    if use_gpu:
        # initial GPU memory(M), device ID
        config.enable_use_gpu(100, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()

    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=1 << 10,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=False)

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = fluid.core.create_paddle_predictor(config)
    return predictor


def load_executor(model_dir, use_gpu=False):
    if use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    program, feed_names, fetch_targets = fluid.io.load_inference_model(
        dirname=model_dir,
        executor=exe,
        model_filename='__model__',
        params_filename='__params__')
    return exe, program, fetch_targets



class Detector():
    """
    Args:
        model_dir (str): root path of __model__, __params__ and infer_cfg.yml
        use_gpu (bool): whether use gpu
    """

    def __init__(self,
                 model_dir,
                 use_gpu=False,
                 run_mode='fluid',
                 threshold=0.5):
        self.config = Config(model_dir)
        if self.config.use_python_inference:
            self.executor, self.program, self.fecth_targets = load_executor(
                model_dir, use_gpu=use_gpu)
        else:
            self.predictor = load_predictor(
                model_dir,
                run_mode=run_mode,
                min_subgraph_size=self.config.min_subgraph_size,
                use_gpu=use_gpu)
        self.preprocess_ops = []
        for op_info in self.config.preprocess_infos:
            op_type = op_info.pop('type')
            self.preprocess_ops.append(eval(op_type)(**op_info))

    def preprocess(self, im):
        # process image by preprocess_ops
        im_info = {
            'scale': [1., 1.],
            'origin_shape': None,
            'resize_shape': None,
        }
        im, im_info = decode_image(im, im_info)
        for operator in self.preprocess_ops:
            im, im_info = operator(im, im_info)
        im = np.array((im, )).astype('float32')
        inputs = create_inputs(im, im_info, self.config.arch)
        return inputs, im_info

    def postprocess(self, boxes, scores, classes, masks, threshold):
        # postprocess output of predictor
        results = {}
        # 再做一次分数过滤
        if threshold > 0.0:
            expect_boxes = scores > threshold
            boxes = boxes[expect_boxes, :]
            scores = scores[expect_boxes]
            classes = classes[expect_boxes]
            masks = masks[expect_boxes, :, :]
        results['boxes'] = boxes
        results['scores'] = scores
        results['classes'] = classes
        results['masks'] = masks
        return results

    def predict(self,
                image,
                threshold=0.5,
                warmup=0,
                repeats=1,
                run_benchmark=False):
        '''
        Args:
            image (str/np.ndarray): path of image/ np.ndarray read by cv2
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape:[N, class_num, mask_resolution, mask_resolution]
        '''
        inputs, im_info = self.preprocess(image)
        np_boxes, np_masks = None, None
        if self.config.use_python_inference:
            for i in range(warmup):
                outs = self.executor.run(self.program,
                                         feed=inputs,
                                         fetch_list=self.fecth_targets,
                                         return_numpy=False)
            t1 = time.time()
            for i in range(repeats):
                outs = self.executor.run(self.program,
                                         feed=inputs,
                                         fetch_list=self.fecth_targets,
                                         return_numpy=False)
            t2 = time.time()
            ms = (t2 - t1) * 1000.0 / repeats
            print("Inference: {} ms per batch image".format(ms))

            np_boxes = np.array(outs[0])
            if self.config.mask_resolution is not None:
                np_masks = np.array(outs[1])
        else:
            input_names = self.predictor.get_input_names()
            for i in range(len(input_names)):
                input_tensor = self.predictor.get_input_tensor(input_names[i])
                input_tensor.copy_from_cpu(inputs[input_names[i]])

            for i in range(warmup):
                self.predictor.zero_copy_run()
                output_names = self.predictor.get_output_names()
                outs0 = self.predictor.get_output_tensor(output_names[0])
                outs1 = self.predictor.get_output_tensor(output_names[1])
                outs2 = self.predictor.get_output_tensor(output_names[2])
                outs0 = outs0.copy_to_cpu()
                outs1 = outs1.copy_to_cpu()
                outs2 = outs2.copy_to_cpu()

            t1 = time.time()
            for i in range(repeats):
                self.predictor.zero_copy_run()
                output_names = self.predictor.get_output_names()
                outs0 = self.predictor.get_output_tensor(output_names[0])
                outs1 = self.predictor.get_output_tensor(output_names[1])
                outs2 = self.predictor.get_output_tensor(output_names[2])
                outs0 = outs0.copy_to_cpu()
                outs1 = outs1.copy_to_cpu()
                outs2 = outs2.copy_to_cpu()
                # 识别哪个输出是masks 哪个输出是scores 哪个输出是classes
                for o in [outs0, outs1, outs2]:
                    if len(o.shape) == 3:
                        masks = o  # [M, img_h, img_w]
                    else:
                        if o.dtype == np.float32:
                            scores = o  # [M, ]
                        else:
                            classes = o  # [M, ]
            t2 = time.time()
            ms = (t2 - t1) * 1000.0 / repeats
            print("Inference: {} ms per batch image".format(ms))

        # do not perform postprocess in benchmark mode
        results = []
        if not run_benchmark:
            if scores[0] < 0.0:
                print('[WARNNING] No object detected.')
                results = {'boxes': np.array([])}
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
                results = self.postprocess(boxes, scores, classes, masks, threshold=threshold)
        return results


def predict_image():
    detector = Detector(
        FLAGS.model_dir, use_gpu=FLAGS.use_gpu, run_mode=FLAGS.run_mode)

    output_dir = FLAGS.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取颜色
    num_classes = len(detector.config.labels)
    colors = get_colors(num_classes * 7)

    if FLAGS.run_benchmark:
        detector.predict(
            FLAGS.image_file,
            FLAGS.threshold,
            warmup=100,
            repeats=100,
            run_benchmark=True)
    else:
        results = detector.predict(FLAGS.image_file, FLAGS.threshold, warmup=30, repeats=100)
        boxes = results['boxes']
        scores = results['scores']
        classes = results['classes']
        masks = results['masks']
        image = cv2.imread(FLAGS.image_file)
        image = draw(image, boxes, scores, classes, masks, detector.config.labels, colors)
        img_name = os.path.split(FLAGS.image_file)[-1]
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, image)


def predict_images():
    detector = Detector(
        FLAGS.model_dir, use_gpu=FLAGS.use_gpu, run_mode=FLAGS.run_mode)

    output_dir = FLAGS.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取颜色
    num_classes = len(detector.config.labels)
    colors = get_colors(num_classes * 7)

    path_dir = os.listdir(FLAGS.image_dir)
    # warm up
    if FLAGS.use_gpu:
        for k, filename in enumerate(path_dir):
            img_path = FLAGS.image_dir + filename
            results = detector.predict(img_path, FLAGS.threshold)
            if k == 10:
                break

    num_imgs = len(path_dir)
    start = time.time()

    for k, filename in enumerate(path_dir):
        img_path = FLAGS.image_dir + filename
        results = detector.predict(img_path, FLAGS.threshold)
        boxes = results['boxes']
        scores = results['scores']
        classes = results['classes']
        masks = results['masks']
        image = cv2.imread(img_path)
        image = draw(image, boxes, scores, classes, masks, detector.config.labels, colors)
        img_name = os.path.split(img_path)[-1]
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, image)

    cost = time.time() - start
    print('total time: {0:.6f}s'.format(cost))
    print('Speed: %.6fs per image,  %.1f FPS.' % ((cost / num_imgs), (num_imgs / cost)))


def predict_video():
    detector = Detector(
        FLAGS.model_dir, use_gpu=FLAGS.use_gpu, run_mode=FLAGS.run_mode)


    output_dir = FLAGS.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取颜色
    num_classes = len(detector.config.labels)
    colors = get_colors(num_classes * 7)


    capture = cv2.VideoCapture(FLAGS.video_file)
    fps = 60
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.split(FLAGS.video_file)[-1]
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    out_path = os.path.join(FLAGS.output_dir, video_name)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    index = 1
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        print('detect frame:%d' % (index))
        index += 1
        results = detector.predict(frame, FLAGS.threshold)


        boxes = results['boxes']
        scores = results['scores']
        classes = results['classes']
        masks = results['masks']
        frame = draw(frame, boxes, scores, classes, masks, detector.config.labels, colors, random_color=False)

        writer.write(frame)
    writer.release()


def play_video():
    detector = Detector(
        FLAGS.model_dir, use_gpu=FLAGS.use_gpu, run_mode=FLAGS.run_mode)


    output_dir = FLAGS.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取颜色
    num_classes = len(detector.config.labels)
    colors = get_colors(num_classes * 7)


    capture = cv2.VideoCapture(FLAGS.play_video)
    fps = 60
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.split(FLAGS.video_file)[-1]
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    out_path = os.path.join(FLAGS.output_dir, video_name)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    index = 1
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        print('detect frame:%d' % (index))
        index += 1
        results = detector.predict(frame, FLAGS.threshold)


        boxes = results['boxes']
        scores = results['scores']
        classes = results['classes']
        masks = results['masks']
        frame = draw(frame, boxes, scores, classes, masks, detector.config.labels, colors, random_color=False)

        cv2.imshow("detection", frame)
        writer.write(frame)
        if cv2.waitKey(110) & 0xff == 27:
            break
    writer.release()


def print_arguments(args):
    print('-----------  Running Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help=("Directory include:'__model__', '__params__', "
              "'infer_cfg.yml', created by tools/export_model.py."),
        required=True)
    parser.add_argument(
        "--image_dir", type=str, default='', help="Path of image dir.")
    parser.add_argument(
        "--image_file", type=str, default='', help="Path of image file.")
    parser.add_argument(
        "--video_file", type=str, default='', help="Path of video file.")
    parser.add_argument(
        "--play_video", type=str, default='', help="Path of video file.")
    parser.add_argument(
        "--run_mode",
        type=str,
        default='fluid',
        help="mode of running(fluid/trt_fp32/trt_fp16)")
    parser.add_argument(
        "--use_gpu",
        type=ast.literal_eval,
        default=False,
        help="Whether to predict with GPU.")
    parser.add_argument(
        "--run_benchmark",
        type=ast.literal_eval,
        default=False,
        help="Whether to predict a image_file repeatedly for benchmark")
    parser.add_argument(
        "--threshold", type=float, default=0.15, help="Threshold of score.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory of output visualization files.")

    FLAGS = parser.parse_args()
    print_arguments(FLAGS)

    if FLAGS.image_file != '' and FLAGS.video_file != '':
        assert "Cannot predict image and video at the same time"
    if FLAGS.image_dir != '':
        predict_images()
    if FLAGS.image_file != '':
        predict_image()
    if FLAGS.video_file != '':
        predict_video()
    if FLAGS.play_video != '':
        play_video()
