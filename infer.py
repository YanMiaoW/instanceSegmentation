import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2 as cv
import os
from ymlib.dataset_visual import crop_pad
from instanceSegmentation.model.segment import Segment


def get_instance_model(checkpoint_path='/Users/yanmiao/yanmiao/checkpoint/not_exist') -> nn.Module:
    instance_model = Segment(3)
    if os.path.exists(checkpoint_path):
        print(f'loading model from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        instance_model.load_state_dict(checkpoint["state_dict"])
    else:
        print('instance_model checkpoint is not found')

    return instance_model


def infer_instance(model: Segment, image: np.ndarray, segment_mask: np.ndarray, rect: list = None, pad=0, bolder=0) -> np.ndarray:
    h, w = image.shape[:2]
    if rect is None:
        rect = [0, 0, w, h]
    x1, y1 = rect[:2]

    image = crop_pad(image, xyxy=rect, bias_xyxy=[-pad, -pad, pad, pad])

    if segment_mask is not None:
        segment_mask = crop_pad(segment_mask, xyxy=rect, bias_xyxy=[-pad, -pad, pad, pad])
        image = np.bitwise_and(image, segment_mask[:, :, np.newaxis])

    if min(*image.shape[:2]) < 50:
        return [], None, None

    if bolder != 0:
        image = cv.copyMakeBorder(image, bolder, bolder, bolder, bolder, cv.BORDER_CONSTANT, value=0)

    # 添加预测
    image = 
    input_ = torch.cat([x, heatmaps], dim=1)
        out = self.forward(inp)
        return torch.sigmoid(out)


    instance_mask = model.test(image, segment_mask)

    if bolder != 0:
        instance_mask = crop_pad(instance_mask, bias_xyxy=[bolder, bolder, -bolder, -bolder])

    x1, y1, x2, y2 = rect
    instance_mask = crop_pad(instance_mask, bias_xyxy=[-x1 + pad, -y1 + pad, w - x2 - pad, h - y2 - pad])

    return instance_mask