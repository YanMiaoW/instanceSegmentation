import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2 as cv
import os
from ymlib.dataset_visual import crop_pad
from instanceSegmentation.model.segment import Segment
import torchvision.transforms as transforms
from functools import partial


def get_instance_model(checkpoint_path='/Users/yanmiao/yanmiao/checkpoint/not_exist') -> nn.Module:
    instance_model = Segment(3)
    if os.path.exists(checkpoint_path):
        print(f'loading model from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        instance_model.load_state_dict(checkpoint["state_dict"])
    else:
        print('instance_model checkpoint is not found')

    return instance_model


def infer_instance(model: Segment,
                   image: np.ndarray,
                   segment_mask: np.ndarray,
                   mask=None,
                   rect: list = None,
                   pad=0,
                   bolder=0) -> np.ndarray:
    h, w = image.shape[:2]
    if rect is None:
        rect = [0, 0, w, h]
    x1, y1 = rect[:2]

    crop_pad_ = partial(crop_pad, xyxy=rect, bias_xyxy=[-pad, -pad, pad, pad])

    image = crop_pad_(image)

    segment_mask = crop_pad_(segment_mask)

    if mask is not None:
        mask = crop_pad_(mask)
        image = np.bitwise_and(image, mask[:, :, np.newaxis])
        segment_mask = np.bitwise_and(segment_mask, mask)

    if min(*image.shape[:2]) < 50:
        return [], None, None

    if bolder != 0:
        image = cv.copyMakeBorder(image, bolder, bolder, bolder, bolder, cv.BORDER_CONSTANT, value=0)
        segment_mask = cv.copyMakeBorder(segment_mask, bolder, bolder, bolder, bolder, cv.BORDER_CONSTANT, value=0)

    # 添加预测
    cut_size = image.shape[:2]
    
    input_size = (480, 480)

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    mask_transform = transforms.ToTensor()
    image = cv.resize(image, input_size)
    segment_mask = cv.resize(segment_mask, input_size)
    image_tensor = image_transform(image)
    # TODO 添加mask训练
    segment_mask_tensor = mask_transform(segment_mask)


    input_tensor = torch.cat([image_tensor], dim=0)
    input_tensor = torch.unsqueeze(input_tensor, axis=0)

    input_tensor = input_tensor.to(next(model.parameters()).device)

    output_tensor = model(input_tensor)
    instance_mask = torch.sigmoid(output_tensor)
    instance_mask = (instance_mask[0][0] * 255).detach().numpy().astype(np.uint8)

    # 恢复mask
    instance_mask = cv.resize(instance_mask, cut_size[::-1])

    if bolder != 0:
        instance_mask = crop_pad(instance_mask, bias_xyxy=[bolder, bolder, -bolder, -bolder])

    x1, y1, x2, y2 = rect
    instance_mask = crop_pad(instance_mask, bias_xyxy=[-x1 + pad, -y1 + pad, w - x2 - pad, h - y2 - pad])

    return instance_mask