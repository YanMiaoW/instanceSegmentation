import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import os
from ymlib.dataset_visual import crop_pad


def get_instance_model(checkpoint_path='/Users/yanmiao/yanmiao/checkpoint/not_exist') -> nn.Module:
    from instanceSegmentation.model.segment import Segment
    instance_model = Segment(3)
    if os.path.exists(checkpoint_path):
        print(f'loading model from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        instance_model.load_state_dict(checkpoint["state_dict"])
    else:
        print('instance_model checkpoint is not found')

    return instance_model


def infer_instance(model, image, mask, rect=None, pad=0, bolder=0):
    h, w = image.shape[:2]
    if rect is None:
        rect = [0, 0, w, h]
    x1, y1 = rect[:2]

    image = crop_pad(image, xyxy=rect, bias_xyxy=[-pad, -pad, pad, pad])

    if mask is not None:
        mask = crop_pad(mask, xyxy=rect, bias_xyxy=[-pad, -pad, pad, pad])
        image = np.bitwise_and(image, mask[:, :, np.newaxis])

    if min(*image.shape[:2]) < 50:
        return [], None, None

    if bolder != 0:
        image = cv.copyMakeBorder(image, bolder, bolder, bolder, bolder, cv.BORDER_CONSTANT, value=0)


    return mask