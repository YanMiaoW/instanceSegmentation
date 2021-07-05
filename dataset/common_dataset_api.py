import torch
import numpy as np
import cv2 as cv
import os
import glob
import json
import random
from imgaug import augmenters as iaa
from typing import AnyStr, Generator

# type
TYPE = {'other', 'sub_dict', 'sub_list', 'class', 'keypoint_status',
        'image_path', 'heatmap_path', 'imencode', 'body_part_name',
        'bitmap',  'mask_path', 'box_xyxy', 'keypoint_xy', 'polygon'}

AUG_TYPE = {'image', 'heatmap', 'mask'}

# key
ANN_CHOICES = {'meta', 'object', 'image', 'mix',
               'class', 'segment_mask', 'class_mask'}

OBJ_CHOICES = {'box', 'class', 'instance_mask', 'body_keypoint'}

KEYPOINT_CHOICES = {'name', 'status', 'point'}

CLS_MASKS_CHOICES = {'class', 'segment_mask'}

# value
KEYPOINT_STATUS = {'missing', 'vis', 'not_vis'}

BODY_PART_NAME = {"head", "neck", "right_shoulder", "right_elbow", "right_wrist",
                  "left_shoulder", "left_elbow", "left_wrist",
                  "right_hip", "right_knee", "right_ankle",
                  "left_hip", "left_knee", "left_ankle", 'right_ear',
                  'left_ear', 'nose', 'right_eye', 'left_eye'}

CLASS = {'person', 'background'}



'''
other类型不会在 transfer 和 aug 阶段进行任何处理，所以其下面不会有类型信息
json条目可以出现缺省, 比如分类数据集没有实例分割，不需要添加分割的条目
json可以有自定义的条目，不过为了多个数据集可以联合操作，在key_combine中强制统一名称 （取消key_combine assert）

{
    other::meta 额外信息
                {
                    origin_image_path 原始图像路径
                    width 图像宽
                    height 图像高
                    ...
                }
    image_path::image 图像
    image_path::mix 结果展示图像
    mask_path::segment_mask 分割掩码（单类别语义分割）
    class:class 类别（分类）
    sub_list::object 个体（实例分割）
        [
            {
                box_xyxy:box 检测框
                class::class 类别
                mask_path::instance_mask 实例分割掩码
                polygon::instance_mask 实例分割掩码polygon（可能）
                sub_list::body_keypoint 人体关键点
                    [
                        {
                            body_part_name::name 人体部位名称
                            keypoint_status::status 关键点状态（缺失，可视，不可视）
                            keypoint_xy::point 关键点坐标xy
                        },
                        ...
                    ]
            },
            ...
        ]
    sub_list::class_mask 多类别语义分割
        [
            {
                class::class 类别
                mask_path::segment_mask 分割掩码
            },
            ...
        ]
}

'''


def key_combine(key: AnyStr, type_: AnyStr) -> AnyStr:
    assert type_ in TYPE or type_ in AUG_TYPE
    assert any(key in s for s in [ANN_CHOICES, OBJ_CHOICES, KEYPOINT_CHOICES,
                                  CLS_MASKS_CHOICES, BODY_PART_NAME])
    return f'{type_}::{key}'


def key_decompose(key: AnyStr) -> AnyStr:
    return key.split('::')[::-1]


def common_ann_loader(dataset_dir: AnyStr, shuffle: bool = False) -> dict:
    ann_paths = glob.glob(os.path.join(dataset_dir, 'data', '*.json'))
    if shuffle:
        ann_paths = random.shuffle(ann_paths)
    else:
        ann_paths = sorted(ann_paths, key=os.path.getmtime)

    for ann_path in ann_paths:
        with open(ann_path) as f:
            ann = json.load(f)

        if 'meta' in ann:
            ann['meta']['dataset_dir'] = dataset_dir

        def path_complete(result):
            for key_type, value in result.items():
                key, type_ = key_decompose(key_type)
                if type_ in ['image_path', 'mask_path', 'heatmap_path']:
                    result[key_type] = os.path.join(dataset_dir, value)
                elif type_ == 'sub_list':
                    for i in value:
                        path_complete(i)
                elif type_ == 'sub_dict':
                    path_complete(value)

        path_complete(ann)

        yield ann


def common_choice(result: dict, key_choices: set = None, type_choices: set = None, key_type_choices: set = None,
                  key_removes: set = None, type_removes: set = None, key_type_removes: set = None) -> dict:
    for key_type in list(result.keys()):
        key, type_ = key_decompose(key_type)

        if key_choices is not None and key not in key_choices:
            del result[key_type]

        if type_choices is not None and type_ not in type_choices:
            del result[key_type]

        if key_type_choices is not None and key_type not in key_type_choices:
            del result[key_type]

        if key_removes is not None and key in key_removes:
            del result[key_type]

        if type_removes is not None and type_ in type_removes:
            del result[key_type]

        if key_type_removes is not None and key_type in key_type_removes:
            del result[key_type]


def common_filter(result: dict, yield_filter: Generator[bool, dict, bool], has_type: bool = False) -> bool:
    ''' yield_filter一定要有yield（生成器）， yield返回的都是true，common_filter才会返回true '''
    if has_type:
        for b in yield_filter(result):
            if not b:
                return False
        return True
    else:
        no_type_result = result.copy()

        def remove_type(result: dict) -> None:
            for key_type, value in result.copy().items():
                key, type_ = key_decompose(key_type)
                if type_ == 'sub_dict':
                    remove_type(value)
                elif type_ == 'sub_list':
                    for i, sub in enumerate(value):
                        remove_type(sub)
                        value[i] = sub
                    result[key] = value
                    del result[key_type]
                else:
                    result[key] = value
                    del result[key_type]

        remove_type(no_type_result)

        for b in yield_filter(no_type_result):
            if not b:
                return False
        return True


def common_transfer(result: dict) -> None:
    for key_type, value in list(result.items()):
        key, type_ = key_decompose(key_type)

        if type_ == 'image_path':
            result[key_combine(key, 'image')] = cv.imread(
                value, cv.IMREAD_COLOR)
            del result[key_type]

        if type_ == 'heatmap_path':
            result[key_combine(key, 'heatmap')] = \
                cv.imread(value, cv.IMREAD_COLOR)
            del result[key_type]

        if type_ == 'mask_path':
            result[key_combine(key, 'mask')] = cv.imread(
                value, cv.IMREAD_GRAYSCALE)
            del result[key_type]

        if type_ == 'class':
            assert value in CLASS

        if type_ == 'keypoint_status':
            assert value in KEYPOINT_STATUS

        if type_ == 'body_part_name':
            assert value in BODY_PART_NAME

        if type_ == 'polygon':
            assert False, 'not support'


def common_aug(result: dict, imgaug: iaa.Augmenter) -> None:
    aug = imgaug._to_deterministic()
    for key_type, value in list(result.items()):
        key, type_ = key_decompose(key_type)

        if type_ == 'image':
            result[key_type] = aug.augment_images([value])[0]

        if type_ == 'mask':
            result[key_type] = aug.augment_images([value])[0]

        if type_ == 'box_xyxy':
            result[key_type] = aug.augment_keypoints(value)[0]

        if type_ == 'keypoint_xy':
            result[key_type] = aug.augment_keypoints(value)[0]


# depreciated 自己用json操作更方便
# def common_merge(*results: dict) -> None:
#     assert len(results) > 1
#     out = {}
#     for result in results[::-1]:
#         for key_type, value in result.items():
#             key, type_ = key_decompose(key_type)

#             if key_type not in out:
#                 out[key_type] = value

#     for key, value in out.items():
#         results[0][key] = value


if __name__ == "__main__":
    from debug_function import *
