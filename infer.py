import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import os
import math
from functools import partial
import torchvision.transforms as transforms

from ymlib.common_dataset_api import key_combine
from ymlib.dataset_util import crop_pad
from ymlib.dataset_visual import index2color
from instanceSegmentation.model.segment import Segment

MODEL_INPUT_SIZE = (480, 480)

ORDER_PART_NAMES = [
    "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", 'right_ear', 'left_ear', 'nose', 'right_eye', 'left_eye'
]

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


def keypoint2heatmaps(keypoint, shape, sigma=10, threshold=0.01):

    r = math.sqrt(math.log(threshold) * (-sigma**2))

    h, w = shape

    heatmaps = np.zeros((h, w, len(ORDER_PART_NAMES)), dtype=np.float32)

    heatmap_show = np.zeros((h, w, 3), dtype=np.uint8)

    for i0, key in enumerate(ORDER_PART_NAMES):

        heatmap = heatmaps[:, :, i0]

        key_type = key_combine(key, 'sub_dict')

        if key_type in keypoint and\
            keypoint[key_type][
                key_combine('status', 'keypoint_status')] != 'missing':

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

    return heatmaps, heatmap_show


def connection2pafs(keypoint, shape, sigma=10):

    h, w = shape

    pafs = np.zeros((h, w, len(CONNECTION_PARTS) * 2), dtype=np.float32)

    paf_show = np.zeros((h, w, 3), dtype=np.uint8)

    for i0, (part1, part2) in enumerate(CONNECTION_PARTS):

        pafx = pafs[:, :, i0 * 2]
        pafy = pafs[:, :, i0 * 2 + 1]

        part1 = key_combine(part1, 'sub_dict')
        part2 = key_combine(part2, 'sub_dict')
        status = key_combine('status', 'keypoint_status')
        point = key_combine('point', 'point_xy')

        if part1 in keypoint and\
                keypoint[part1][status] != 'missing' and\
                part2 in keypoint and\
                keypoint[part2][status]  != 'missing':

            x1, y1 = keypoint[part1][point]
            x2, y2 = keypoint[part2][point]

            v0 = np.array([x2 - x1, y2 - y1])
            v0_norm = np.linalg.norm(v0)

            if v0_norm == 0:
                continue

            if x1 > 0 and x1 < w and y1 > 0 and y1 < h and\
                x2 > 0 and x2 < w and y2 > 0 and y2 < h:

                x_min = max(int(round(min(x1, x2) - sigma)), 0)
                x_max = min(int(round(max(x1, x2) + sigma)), w)
                y_min = max(int(round(min(y1, y2) - sigma)), 0)
                y_max = min(int(round(max(y1, y2) + sigma)), h)

                xs, ys = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))

                vs = np.stack((xs.flatten() - x1, ys.flatten() - y1), axis=1)

                # |B|cos = |A||B|cos / |A|
                l = np.dot(vs, v0) / v0_norm
                c1 = (l >= 0) & (l <= v0_norm)

                # |B|sin = |A||B|sin / |A|
                c2 = abs(np.cross(vs, v0) / v0_norm) <= sigma

                idxs = c1 & c2

                idxs = idxs.reshape(xs.shape)

                pafx[y_min:y_max, x_min:x_max][idxs] = v0[0] / v0_norm
                pafy[y_min:y_max, x_min:x_max][idxs] = v0[1] / v0_norm

                show_region = paf_show[y_min:y_max, x_min:x_max]
                color_region = np.zeros((y_max - y_min, x_max - x_min, 3), np.float32)
                color_region[idxs] = index2color(i0, len(CONNECTION_PARTS))
                show_region[:] = np.max(np.stack((show_region, color_region)), axis=0)

    return pafs, paf_show


def get_instance_model(checkpoint_path='/Users/yanmiao/yanmiao/checkpoint/instanceSegmentation/final_best_0727.pth') -> nn.Module:
    instance_model = Segment(3 + 1 + len(ORDER_PART_NAMES) + len(CONNECTION_PARTS) * 2)
    if os.path.exists(checkpoint_path):
        print(f'loading model from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        instance_model.load_state_dict(checkpoint["state_dict"])
    else:
        print('instance_model checkpoint is not found')

    return instance_model


def infer_instance(model: Segment,
                   image: np.ndarray,
                   segment_mask: np.ndarray,
                   keypoints:dict,
                   mask=None,
                   rect: list = None,
                   pad=0,
                   border=0) -> np.ndarray:

    h, w = image.shape[:2]
    if rect is None:
        rect = [0, 0, w, h]
    x1, y1 = rect[:2]

    crop_pad_ = partial(crop_pad, xyxy=rect, ltrb=[pad, pad, pad, pad])

    image = crop_pad_(image)

    segment_mask = crop_pad_(segment_mask)

    if mask is not None:
        mask = crop_pad_(mask)
        image = np.bitwise_and(image, mask[:, :, np.newaxis])
        segment_mask = np.bitwise_and(segment_mask, mask)

    if min(*image.shape[:2]) < 50:
        return [], None, None

    if border != 0:
        copyMakeBorder_ = partial(cv.copyMakeBorder,
                                  top=border,
                                  bottom=border,
                                  left=border,
                                  right=border,
                                  borderType=cv.BORDER_CONSTANT,
                                  value=0)
        image = copyMakeBorder_(image)
        segment_mask = copyMakeBorder_(segment_mask)

    # 添加预测
    pre_size = image.shape[:2]

    normal = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    to_tensor = transforms.ToTensor()

    image = cv.resize(image, MODEL_INPUT_SIZE)
    segment_mask = cv.resize(segment_mask, MODEL_INPUT_SIZE)

    heatmaps, heatmap_show = keypoint2heatmaps(keypoints, MODEL_INPUT_SIZE)
    pafs, paf_show = connection2pafs(keypoints, MODEL_INPUT_SIZE)

    image_tensor = normal(to_tensor(image))
    segment_mask_tensor = to_tensor(segment_mask)

    heatmap_tensors = [to_tensor(heatmap) for heatmap in heatmaps]
    heatmap_tensor = torch.cat(heatmap_tensors, dim=0)

    paf_tensors = [to_tensor(paf) for paf in pafs]
    paf_tensor = torch.cat(paf_tensors, dim=0)

    input_tensor = torch.cat([image_tensor,segment_mask_tensor, heatmap_tensor, paf_tensor], dim=0)
    input_tensor = torch.unsqueeze(input_tensor, axis=0)

    input_tensor = input_tensor.to(next(model.parameters()).device)

    instance_mask_tensor = torch.sigmoid(model(input_tensor))
    instance_mask = (instance_mask_tensor[0][0] * 255).detach().cpu().numpy().astype(np.uint8)

    # 恢复mask
    instance_mask = cv.resize(instance_mask, pre_size[::-1])

    if border != 0:
        instance_mask = crop_pad(instance_mask, ltrb=[-border, -border, -border, -border])

    x1, y1, x2, y2 = rect
    instance_mask = crop_pad(instance_mask, ltrb=[x1 - pad, y1 - pad, w - x2 - pad, h - y2 - pad])

    return instance_mask
