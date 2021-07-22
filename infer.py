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

CONNECTION_PARTS = [
    ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
    ['right_hip', 'right_knee'],
    ['right_knee', 'right_ankle'],
    ['right_ear', 'nose'],
    ['right_eye', 'nose'],
    ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'],
    ['left_hip', 'left_knee'],
    ['left_knee', 'left_ankle'],
    ['left_ear', 'nose'],
    ['left_eye', 'nose'],
    ['right_ear', 'left_eye'],
]


def connection2pafs(keypoint, shape, sigma=10):

    pafs = []

    paf_show = np.zeros((*shape, 3), dtype=np.uint8)

    for i0, (part1, part2) in enumerate(CONNECTION_PARTS):

        pafx = np.zeros(shape, dtype=np.float32)
        pafy = np.zeros(shape, dtype=np.float32)

        part1 = key_combine(part1, 'sub_dict')
        part2 = key_combine(part2, 'sub_dict')
        status = key_combine('status', 'keypoint_status')
        point = key_combine('point', 'point_xy')

        if part1 in keypoint and\
                keypoint[part1][status] == 'vis' and\
                part2 in keypoint and\
                keypoint[part2][status] == 'vis':

            x1, y1 = keypoint[part1][point]
            x2, y2 = keypoint[part2][point]

            v0 = np.array([x2 - x1, y2 - y1])
            v0_norm = np.linalg.norm(v0)

            if v0_norm == 0:
                continue

            h, w = shape
            x_min = max(int(round(min(x1, x2) - sigma)), 0)
            x_max = min(int(round(max(x1, x2) + sigma)), w - 1)
            y_min = max(int(round(min(y1, y2) - sigma)), 0)
            y_max = min(int(round(max(y1, y2) + sigma)), h - 1)

            xs, ys = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))

            vs = np.stack((xs.flatten() - x1, ys.flatten() - y1), axis=1)

            # |B|cos = |A||B|cos / |A|
            l = np.dot(vs, v0) / v0_norm
            c1 = (l >= 0) & (l <= v0_norm)

            # |B|sin = |A||B|sin / |A|
            c2 = abs(np.cross(vs, v0) / v0_norm) <= sigma

            idxs = c1 & c2

            idxs = idxs.reshape(xs.shape)

            region = pafx[y_min:y_max, x_min:x_max]
            region[idxs] = v0[0] / v0_norm
            region[idxs] = v0[1] / v0_norm

            show_region = paf_show[y_min:y_max, x_min:x_max]
            color_region = np.zeros((*region.shape, 3), np.float32)
            color_region[idxs] = index2color(i0, len(CONNECTION_PARTS))
            show_region[:] = np.max(np.stack((show_region, color_region)), axis=0)

        pafs.append(pafx)
        pafs.append(pafy)

    return pafs, paf_show


def get_instance_model(checkpoint_path='/Users/yanmiao/yanmiao/checkpoint/not_exist') -> nn.Module:
    instance_model = Segment(3 + len(CONNECTION_PARTS) * 2)
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
                   pafs: list,
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

    pafs = [crop_pad_(paf) for paf in pafs]

    if mask is not None:
        mask = crop_pad_(mask)
        image = np.bitwise_and(image, mask[:, :, np.newaxis])
        segment_mask = np.bitwise_and(segment_mask, mask)
        pafs = [np.bitwise_and(paf, mask) for paf in pafs]

    if min(*image.shape[:2]) < 50:
        return [], None, None

    if bolder != 0:
        image = cv.copyMakeBorder(image, bolder, bolder, bolder, bolder, cv.BORDER_CONSTANT, value=0)

    # 添加预测
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    mask_transform = transforms.ToTensor()
    paf_transfrom = transforms.ToTensor()

    image_tensor = image_transform(image)
    # TODO 添加mask训练
    segment_mask_tensor = mask_transform(segment_mask)

    paf_tensors = [paf_transfrom(paf) for paf in pafs]
    paf_tensor = torch.cat(paf_tensors, dim=0)
    input_tensor = torch.cat([image_tensor, paf_tensor], dim=1)

    input_tensor = input_tensor.to(model.device)

    output_tensor = model(input_tensor)
    instance_mask = torch.sigmoid(output_tensor)

    # mask转换
    if bolder != 0:
        instance_mask = crop_pad(instance_mask, bias_xyxy=[bolder, bolder, -bolder, -bolder])

    x1, y1, x2, y2 = rect
    instance_mask = crop_pad(instance_mask, bias_xyxy=[-x1 + pad, -y1 + pad, w - x2 - pad, h - y2 - pad])

    return instance_mask