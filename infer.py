import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2 as cv
import os
import math
from ymlib.common_dataset_api import key_combine
from ymlib.dataset_visual import crop_pad, index2color
from instanceSegmentation.model.segment import Segment
import torchvision.transforms as transforms
from functools import partial

ORDER_PART_NAMES = [
    "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", 'right_ear', 'left_ear', 'nose', 'right_eye', 'left_eye'
]


def keypoint2heatmaps(keypoint, shape, sigma=10, threshold=0.01):

    r = math.sqrt(math.log(threshold) * (-sigma**2))

    heatmaps = []

    heatmap_show = np.zeros((*shape, 3), dtype=np.uint8)

    for i0, key in enumerate(ORDER_PART_NAMES):

        heatmap = np.zeros(shape, dtype=np.float32)

        key_type = key_combine(key, 'sub_dict')

        if key_type in keypoint and\
            keypoint[key_type][
                key_combine('status', 'keypoint_status')] == 'vis':

            x, y = keypoint[key_type][key_combine('point', 'point_xy')]
            h, w = shape

            if x > 0 and x < w and y > 0 and y < h:

                x_min = max(0, int(x - r))
                x_max = min(w, int(x + r + 1))
                y_min = max(0, int(y - r))
                y_max = min(h, int(y + r + 1))

                xs = np.arange(x_min, x_max)
                ys = np.arange(y_min, y_max)[:, np.newaxis]

                e_table = np.exp(-((xs - x)**2 + (ys - y)**2) / sigma**2)

                idxs = np.where(e_table > threshold)

                region = heatmap[y_min:y_max, x_min:x_max]
                region[idxs] = e_table[idxs]

                show_region = heatmap_show[y_min:y_max, x_min:x_max]
                color_region = np.zeros((*region.shape, 3), np.float32)
                color_region[:] = index2color(i0, len(ORDER_PART_NAMES))
                color_region = (e_table[:, :, np.newaxis] * color_region).astype(np.uint8)
                show_region[:] = np.max(np.stack((show_region, color_region)), axis=0)

        heatmaps.append(heatmap)

    return heatmaps, heatmap_show


def get_instance_model(checkpoint_path='/Users/yanmiao/yanmiao/checkpoint/not_exist') -> nn.Module:
    instance_model = Segment(3 + len(ORDER_PART_NAMES))
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
                   heatmaps: list,
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

    heatmaps = [crop_pad_(heatmap) for heatmap in heatmaps]

    if mask is not None:
        mask = crop_pad_(mask)
        image = np.bitwise_and(image, mask[:, :, np.newaxis])
        segment_mask = np.bitwise_and(segment_mask, mask)
        heatmaps = [np.bitwise_and(heatmap, mask) for heatmap in heatmaps]

    if min(*image.shape[:2]) < 50:
        return [], None, None

    if bolder != 0:
        copyMakeBorder_ = partial(cv.copyMakeBorder,
                                  top=bolder,
                                  bottom=bolder,
                                  left=bolder,
                                  right=bolder,
                                  borderType=cv.BORDER_CONSTANT,
                                  value=0)
        image = copyMakeBorder_(image)
        segment_mask = copyMakeBorder_(segment_mask)
        heatmaps = [copyMakeBorder_(heatmap) for heatmap in heatmaps]

    # 添加预测
    cut_size = image.shape[:2]

    input_size = (480, 480)

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    mask_transform = transforms.ToTensor()
    heatmap_transfrom = transforms.ToTensor()

    image = cv.resize(image, input_size)
    segment_mask = cv.resize(segment_mask, input_size)
    heatmaps = [cv.resize(heatmap, input_size) for heatmap in heatmaps]

    image_tensor = image_transform(image)
    # TODO 添加mask训练
    segment_mask_tensor = mask_transform(segment_mask)

    heatmap_tensors = [heatmap_transfrom(heatmap) for heatmap in heatmaps]
    heatmap_tensor = torch.cat(heatmap_tensors, dim=0)

    input_tensor = torch.cat([image_tensor, heatmap_tensor], dim=0)
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