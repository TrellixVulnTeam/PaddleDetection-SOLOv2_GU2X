#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-09-09 11:20:53
#   Description : paddle_solov2
#
# ================================================================
import torch
from paddle import fluid
from ppdet.core.workspace import load_config, create


def load_weights(path):
    """ Loads weights from a compressed save file. """
    # state_dict = torch.load(path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict

state_dict = load_weights('SOLOv2_R50_3x.pth')
print('============================================================')

backbone_dic = {}
fpn_dic = {}
head_dic = {}
mask_feat_head_dic = {}
others = {}
for key, value in state_dict['state_dict'].items():
    if 'tracked' in key:
        continue
    if 'backbone' in key:
        backbone_dic[key] = value.data.numpy()
    elif 'neck' in key:
        fpn_dic[key] = value.data.numpy()
    elif 'bbox_head' in key:
        head_dic[key] = value.data.numpy()
    elif 'mask_feat_head' in key:
        mask_feat_head_dic[key] = value.data.numpy()
    else:
        others[key] = value.data.numpy()

print()


cfg = load_config('configs/solov2/solov2_r50_fpn_8gpu_3x.yml')

model = create(cfg.architecture)
inputs_def = cfg['TestReader']['inputs_def']
inputs_def['iterable'] = True
feed_vars, loader = model.build_inputs(**inputs_def)
test_fetches = model.test(feed_vars)


# Create an executor using CPU as an example
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())



print('\nCopying...')


def copy(name, w):
    tensor = fluid.global_scope().find_var(name).get_tensor()
    tensor.set(w, place)

def copy_conv_bn_to_PaddleDetection_ResNet(conv_name, w, scale, offset, m, v):
    bn_name = 'bn' + conv_name[3:]
    tensor = fluid.global_scope().find_var(conv_name + '_weights').get_tensor()
    tensor2 = fluid.global_scope().find_var(bn_name + '_scale').get_tensor()
    tensor3 = fluid.global_scope().find_var(bn_name + '_offset').get_tensor()
    tensor4 = fluid.global_scope().find_var(bn_name + '_mean').get_tensor()
    tensor5 = fluid.global_scope().find_var(bn_name + '_variance').get_tensor()
    tensor.set(w, place)
    tensor2.set(scale, place)
    tensor3.set(offset, place)
    tensor4.set(m, place)
    tensor5.set(v, place)

def copy_conv_to_PaddleDetection_FPN(conv_name, w, b):
    tensor = fluid.global_scope().find_var(conv_name + '_w').get_tensor()
    tensor2 = fluid.global_scope().find_var(conv_name + '_b').get_tensor()
    tensor.set(w, place)
    tensor2.set(b, place)

def copy_conv_bn(conv_name, w, scale, offset, m, v):
    tensor = fluid.global_scope().find_var('%s.conv.weight' % conv_name).get_tensor()
    tensor2 = fluid.global_scope().find_var('%s.bn.scale' % conv_name).get_tensor()
    tensor3 = fluid.global_scope().find_var('%s.bn.offset' % conv_name).get_tensor()
    tensor4 = fluid.global_scope().find_var('%s.bn.mean' % conv_name).get_tensor()
    tensor5 = fluid.global_scope().find_var('%s.bn.var' % conv_name).get_tensor()
    tensor.set(w, place)
    tensor2.set(scale, place)
    tensor3.set(offset, place)
    tensor4.set(m, place)
    tensor5.set(v, place)

def copy_conv_af(conv_name, w, scale, offset):
    tensor = fluid.global_scope().find_var('%s.conv.weight' % conv_name).get_tensor()
    tensor2 = fluid.global_scope().find_var('%s.af.scale' % conv_name).get_tensor()
    tensor3 = fluid.global_scope().find_var('%s.af.offset' % conv_name).get_tensor()
    tensor.set(w, place)
    tensor2.set(scale, place)
    tensor3.set(offset, place)

def copy_conv(conv_name, w, b):
    tensor = fluid.global_scope().find_var('%s.conv.weight' % conv_name).get_tensor()
    tensor2 = fluid.global_scope().find_var('%s.conv.bias' % conv_name).get_tensor()
    tensor.set(w, place)
    tensor2.set(b, place)

def copy_conv_gn(conv_name, w, b, scale, offset):
    tensor = fluid.global_scope().find_var('%s.conv.weight' % conv_name).get_tensor()
    if b is not None:
        tensor2 = fluid.global_scope().find_var('%s.conv.bias' % conv_name).get_tensor()
        tensor2.set(b, place)
    tensor3 = fluid.global_scope().find_var('%s.gn.scale' % conv_name).get_tensor()
    tensor4 = fluid.global_scope().find_var('%s.gn.offset' % conv_name).get_tensor()
    tensor.set(w, place)
    tensor3.set(scale, place)
    tensor4.set(offset, place)


# 获取SOLOv2模型的权重

# Resnet50
w = backbone_dic['backbone.conv1.weight']
scale = backbone_dic['backbone.bn1.weight']
offset = backbone_dic['backbone.bn1.bias']
m = backbone_dic['backbone.bn1.running_mean']
v = backbone_dic['backbone.bn1.running_var']
copy('conv1_weights', w)
copy('bn_conv1_scale', scale)
copy('bn_conv1_offset', offset)
copy('bn_conv1_mean', m)
copy('bn_conv1_variance', v)


nums = [3, 4, 6, 3]
for nid, num in enumerate(nums):
    stage_name = 'res' + str(nid + 2)
    for kk in range(num):
        block_name = stage_name + chr(ord("a") + kk)

        conv_name1 = 'backbone.layer%d.%d' % ((nid+1), kk)
        w = backbone_dic[conv_name1 + '.conv1.weight']
        scale = backbone_dic[conv_name1 + '.bn1.weight']
        offset = backbone_dic[conv_name1 + '.bn1.bias']
        m = backbone_dic[conv_name1 + '.bn1.running_mean']
        v = backbone_dic[conv_name1 + '.bn1.running_var']
        copy_conv_bn_to_PaddleDetection_ResNet(block_name + "_branch2a", w, scale, offset, m, v)

        conv_name2 = 'backbone.layer%d.%d' % ((nid+1), kk)
        w = backbone_dic[conv_name2 + '.conv2.weight']
        scale = backbone_dic[conv_name2 + '.bn2.weight']
        offset = backbone_dic[conv_name2 + '.bn2.bias']
        m = backbone_dic[conv_name2 + '.bn2.running_mean']
        v = backbone_dic[conv_name2 + '.bn2.running_var']
        copy_conv_bn_to_PaddleDetection_ResNet(block_name + "_branch2b", w, scale, offset, m, v)

        conv_name3 = 'backbone.layer%d.%d' % ((nid+1), kk)
        w = backbone_dic[conv_name3 + '.conv3.weight']
        scale = backbone_dic[conv_name3 + '.bn3.weight']
        offset = backbone_dic[conv_name3 + '.bn3.bias']
        m = backbone_dic[conv_name3 + '.bn3.running_mean']
        v = backbone_dic[conv_name3 + '.bn3.running_var']
        copy_conv_bn_to_PaddleDetection_ResNet(block_name + "_branch2c", w, scale, offset, m, v)

        # 每个stage的第一个卷积块才有4个卷积层
        if kk == 0:
            shortcut_name = 'backbone.layer%d.%d.downsample' % ((nid + 1), kk)
            w = backbone_dic[shortcut_name + '.0.weight']
            scale = backbone_dic[shortcut_name + '.1.weight']
            offset = backbone_dic[shortcut_name + '.1.bias']
            m = backbone_dic[shortcut_name + '.1.running_mean']
            v = backbone_dic[shortcut_name + '.1.running_var']
            copy_conv_bn_to_PaddleDetection_ResNet(block_name + "_branch1", w, scale, offset, m, v)
# fpn
w = fpn_dic['neck.lateral_convs.3.conv.weight']
b = fpn_dic['neck.lateral_convs.3.conv.bias']
copy_conv_to_PaddleDetection_FPN('fpn_inner_res5_sum', w, b)

w = fpn_dic['neck.lateral_convs.2.conv.weight']
b = fpn_dic['neck.lateral_convs.2.conv.bias']
copy_conv_to_PaddleDetection_FPN('fpn_inner_res4_sum_lateral', w, b)

w = fpn_dic['neck.lateral_convs.1.conv.weight']
b = fpn_dic['neck.lateral_convs.1.conv.bias']
copy_conv_to_PaddleDetection_FPN('fpn_inner_res3_sum_lateral', w, b)

w = fpn_dic['neck.lateral_convs.0.conv.weight']
b = fpn_dic['neck.lateral_convs.0.conv.bias']
copy_conv_to_PaddleDetection_FPN('fpn_inner_res2_sum_lateral', w, b)

w = fpn_dic['neck.fpn_convs.3.conv.weight']
b = fpn_dic['neck.fpn_convs.3.conv.bias']
copy_conv_to_PaddleDetection_FPN('fpn_res5_sum', w, b)

w = fpn_dic['neck.fpn_convs.2.conv.weight']
b = fpn_dic['neck.fpn_convs.2.conv.bias']
copy_conv_to_PaddleDetection_FPN('fpn_res4_sum', w, b)

w = fpn_dic['neck.fpn_convs.1.conv.weight']
b = fpn_dic['neck.fpn_convs.1.conv.bias']
copy_conv_to_PaddleDetection_FPN('fpn_res3_sum', w, b)

w = fpn_dic['neck.fpn_convs.0.conv.weight']
b = fpn_dic['neck.fpn_convs.0.conv.bias']
copy_conv_to_PaddleDetection_FPN('fpn_res2_sum', w, b)


# head
num_convs = 4
for lvl in range(0, num_convs):
    # conv + gn
    w = head_dic['bbox_head.kernel_convs.%d.conv.weight'%lvl]
    scale = head_dic['bbox_head.kernel_convs.%d.gn.weight'%lvl]
    offset = head_dic['bbox_head.kernel_convs.%d.gn.bias'%lvl]
    copy_conv_gn('head.krn_convs.%d' % (lvl, ), w, None, scale, offset)

    # conv + gn
    w = head_dic['bbox_head.cate_convs.%d.conv.weight'%lvl]
    scale = head_dic['bbox_head.cate_convs.%d.gn.weight'%lvl]
    offset = head_dic['bbox_head.cate_convs.%d.gn.bias'%lvl]
    copy_conv_gn('head.cls_convs.%d' % (lvl, ), w, None, scale, offset)

# 类别分支最后的conv
w = head_dic['bbox_head.solo_cate.weight']
b = head_dic['bbox_head.solo_cate.bias']
copy_conv('head.cls_convs.%d' % (num_convs,), w, b)

# 卷积核分支最后的conv
w = head_dic['bbox_head.solo_kernel.weight']
b = head_dic['bbox_head.solo_kernel.bias']
copy_conv('head.krn_convs.%d' % (num_convs,), w, b)


# mask_feat_head
start_level = 0
end_level = 3
for i in range(start_level, end_level + 1):
    if i == 0:
        w = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv0.conv.weight' % (i,)]
        scale = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv0.gn.weight' % (i,)]
        offset = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv0.gn.bias' % (i,)]
        copy_conv_gn('mask_feat_head.convs_all_levels.%d.conv0' % (i,), w, None, scale, offset)
        continue

    for j in range(i):
        w = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv%d.conv.weight' % (i, j)]
        scale = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv%d.gn.weight' % (i, j)]
        offset = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv%d.gn.bias' % (i, j)]
        copy_conv_gn('mask_feat_head.convs_all_levels.%d.conv%d' % (i, j), w, None, scale, offset)


w = mask_feat_head_dic['mask_feat_head.conv_pred.0.conv.weight']
scale = mask_feat_head_dic['mask_feat_head.conv_pred.0.gn.weight']
offset = mask_feat_head_dic['mask_feat_head.conv_pred.0.gn.bias']
copy_conv_gn('mask_feat_head.conv_pred.0', w, None, scale, offset)



import os
if not os.path.exists('output/'): os.mkdir('output/')
if not os.path.exists('output/solov2_r50_fpn_8gpu_3x/'): os.mkdir('output/solov2_r50_fpn_8gpu_3x/')
fluid.save(fluid.default_startup_program(), 'output/solov2_r50_fpn_8gpu_3x/model_final')
print('\nDone.')


