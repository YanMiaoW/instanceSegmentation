import torch
import numpy as np
import cv2 as cv
import os
import glob
import json
from torch.utils.data import Dataset
import random
from imgaug import augmenters as iaa
from typing import AnyStr, Callable, Generator

TYPE = {'other', 'sub_dict', 'sub_list', 'image_path', 'heatmap_path', 'imencode',
        'bitmap',  'mask_path', 'keypoint', 'box_xyxy', 'polygon'}

AUG_TYPE = {'image', 'heatmap', 'mask'}

ANN_CHOICES = {'meta', 'object', 'image', 'mix',
               'class', 'segment_mask', 'class_mask'}

OBJ_CHOICES = {'box', 'class', 'instance_mask'}

CLS_MASKS_CHOICES = {'class', 'segment_mask'}

CLASS = {'person', 'background'}


def key_combine(key: AnyStr, type_: AnyStr) -> AnyStr:
    assert type_ in TYPE or type_ in AUG_TYPE
    assert any(key in s for s in [ANN_CHOICES, OBJ_CHOICES, CLS_MASKS_CHOICES])
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
    if has_type:
        for b in yield_filter(result):
            if not b:
                return False
        return True
    else:       
        no_type_result = result.copy()
        
        def remove_type(result:dict)->None:
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

        if type_ == 'keypoint':
            result[key_type] = aug.augment_keypoints(value)[0]

def common_merge(*results: dict) -> None:
    assert len(results) > 1
    out = {}
    for result in results[::-1]:
        for key_type, value in result.items():
            key, type_ = key_decompose(key_type)

            if key_type not in out:
                out[key_type] = value

    for key, value in out.items():
        results[0][key] = value
        
        



if __name__ == "__main__":
    from debug_function import *

    class CommonSegmentSingleDataset(Dataset):
        def __init__(self, dataset_dir: AnyStr, imgaug: iaa.Augmenter = None,
                     filter: Callable = None) -> dict:
            super().__init__()

            self.results = []

            for result in common_ann_loader(dataset_dir):
                common_choice(result, key_choices={
                    'image', 'mix', 'segment_mask', 'meta'})
                if common_filter(result, filter):
                    self.results.append(result)

                # sub=result.copy()
                # common_choice(sub,key_choices='image')
                # common_transfer(sub)
                # result = common_merge(sub,result)

            self.imgaug = imgaug

        def __getitem__(self, index):
            result = self.results[index]

            # for 4 in self.results:
            #     np.concatenate
            #     result

            common_transfer(result)
            common_aug(result)
            common_choice(result, type_removes=TYPE - {'sub_dict', 'other'})

            return result

        def __len__(self):
            return len(self.results)

    class CommonClassificationDataset(Dataset):
        def __init__(self, dataset_dir: AnyStr, imgaug: iaa.Augmenter = None,
                     filter: Callable = None) -> dict:
            super().__init__()

            def filter(result):
                yield 'object' in result

                objs = result['object']

                yield any(obj['class'] in ['person', 'background'] for obj in objs)

                for i, obj in enumerate(objs.copy()):
                    if obj['class'] not in ['person', 'background']:
                        del objs[i]

            self.results = []
            for result in common_ann_loader(dataset_dir):
                if 'image' in result and 'class' in result:
                    common_choice(result, key_choices={
                                  'image', 'class', 'meta'})
                    if common_filter(result, filter):
                        self.results.append(result)

            self.imgaug = imgaug

        def __getitem__(self, index):
            result = self.results[index]

            common_transfer(result)
            common_aug(result)
            common_choice(result, type_removes=TYPE - {'sub_dict', 'other'})

            return result

        def __len__(self):
            return len(self.results)

    class CommonInstanceDataset(Dataset):

        def __init__(self, dataset_dir: AnyStr, imgaug: iaa.Augmenter = None,
                     filter_ann: Callable = None, filter_obj: Callable = None,
                     ann_choices: set = {'meta', 'object', 'image', 'mix'},
                     obj_choices: set = OBJ_CHOICES) -> None:
            super().__init__()

            assert all(key in ANN_CHOICES for key in ann_choices)
            assert all(key in OBJ_CHOICES for key in obj_choices)

            self.results = []
            for result in common_ann_loader(dataset_dir):
                if 'object' not in result:
                    continue

                obj = result['object']

                common_choice(result, key_choices=ann_choices)
                common_choice(obj, key_choices=obj_choices)

                if not common_filter(result, filter_ann):
                    continue

                if not common_filter(obj, filter_obj):
                    continue

                self.results.append(result)

            self.imgaug = imgaug

        def __getitem__(self, index):
            result = self.results[index]

            common_transfer(result)
            common_aug(result, self.imgaug)
            common_choice(result, type_removes=TYPE - {'sub_dict', 'other'})

            obj = result['object']

            common_transfer(obj)
            common_aug(obj, self.imgaug)
            common_choice(obj, type_removes=TYPE - {'sub_dict', 'other'})

            return result

        def __len__(self):
            return len(self.results)

    def filter_ann(ann):
        if len(ann['object']) > 1:
            return True

        return False

    def filter_obj(obj):
        if len(obj['box']) == 4:
            return True

        return False

    # for a in comm('/Users/yanmiao/yanmiao/data-common/supervisely',
    #               filter_ann=filter_ann, filter_obj=filter_obj,
    #               ann_choices={'image', 'segment_mask',
    #                            'mix', 'meta'},
    #               obj_choices={'box', 'instance_mask'}):
    #     print(a)
