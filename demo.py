import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import tqdm
import argparse
import glob
import sys

from ymlib.debug_function import *
from ymlib.dataset_visual import mask2box, draw_keypoint, draw_mask, index2color, draw_box, xywh2xyxy
from ymlib.common import path_decompose, get_minimum_memory_footprint_id, get_git_branch_name

from tsai.face import get_face_detect_model, infer_face_detect
from tsai.segment import get_segment_model, infer_segment
from other.LightPoseEstimation.infer import get_pose_model, infer_pose
from instanceSegmentation.infer import get_instance_model, infer_instance


def parse_args():
    parser = argparse.ArgumentParser(description='demo image')
    parser.add_argument('-i', '--image-dir', help='image test dir', required=True)
    parser.add_argument('-o', '--save-dir', help='output save dir', required=True)
    parser.add_argument('--checkpoint-dir', help='checkpoint save dir')
    parser.add_argument('--checkpoint-path', help='checkpoint save path')
    parser.add_argument('--continue-test', action='store_true', help='skip existing file.')
    parser.add_argument('--auto-gpu-id', action='store_true', help='gpu inference')

    args = parser.parse_args()
    return args


def contour2mask(contours, hierarchy, index, shape):
    def get_root_and_level(index, level_count=0):
        parent = hierarchy[0][index][3]
        if parent == -1:
            return index, level_count

        return get_root_and_level(parent, level_count + 1)

    mask = np.zeros(shape, dtype=np.uint8)
    for j0, _ in enumerate(contours):
        root, level = get_root_and_level(j0)
        if root != index:
            continue

        color = (255, 255, 255) if level % 2 == 0 else (0, 0, 0)
        cv.drawContours(mask, contours, j0, color, thickness=-1)

    return mask


if __name__ == "__main__":
    args = parse_args()

    branch_name = get_git_branch_name()

    # 加载模型
    if args.checkpoint_path is not None:
        branch_best_path = args.checkpoint_path
    elif args.checkpoint_dir is not None:
        branch_best_path = os.path.join(args.checkpoint_dir, f'{branch_name}_best.pth')
    else:
        branch_best_path = ''

    instance_model = get_instance_model(branch_best_path)

    pose_model = get_pose_model()
    segment_model = get_segment_model()
    face_detect_model = get_face_detect_model()

    # 加载到显存 segment 不支持gpu
    if args.auto_gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        device = torch.device(f"cuda:{get_minimum_memory_footprint_id()}")
    else:
        device = 'cpu'

    instance_model = instance_model.to(device)
    pose_model = pose_model.to(device)

    face_detect_model = face_detect_model.to(device)

    # 遍历预测图片
    print(f'loading image from {args.image_dir}')
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*[jpg,png,jpgerr]")))

    save_dir = os.path.join(args.save_dir, branch_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f'save result mix to {save_dir}')

    for i0, filepath in enumerate(tqdm.tqdm(image_paths)):
        if i0 != 13:
            continue
        # filepath = '/Users/yanmiao/yanmiao/data-common/supervisely/image/05411.png'

        _, basename, _ = path_decompose(filepath)
        result_path = os.path.join(save_dir, f'{basename}.jpg')

        if args.continue_test and os.path.exists(result_path):
            continue

        image = cv.imread(filepath)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        mix = image.copy()

        segment_mask = infer_segment(segment_model, image)
        # imshow(segment_mask)

        # segment_path = '/Users/yanmiao/yanmiao/data-common/supervisely/segment_mask/05411.png'
        # segment_mask = cv.imread(segment_path, cv.IMREAD_GRAYSCALE)

        _, segment_mask_binary = cv.threshold(segment_mask, 127, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(segment_mask_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        instance_num = (hierarchy[0, :, 3] == -1).sum()

        k0 = 0

        #TODO contours封装成api
        for j0, (contour, parent) in enumerate(zip(contours, hierarchy[0, :, 3])):
            # 只处理最外层
            if parent != -1:
                continue

            # 获取最外侧外接框
            rect = cv.boundingRect(contour)
            rw, rh = rect[2:]

            if rw < w * 0.1 or rh < h * 0.1:
                continue

            rect = xywh2xyxy(rect)

            draw_box(mix, rect)

            faces = infer_face_detect(face_detect_model, image, mask=segment_mask, rect=rect, bolder=16)

            if len(faces) <= 1:
                # 获取mask
                mask = contour2mask(contours, hierarchy, j0, segment_mask.shape)

                draw_mask(mix, mask, index2color(j0, instance_num))

                for k0, box_xyxy in enumerate(faces):
                    draw_box(mix, box_xyxy, index2color(k0, len(faces)))

                poses, _, _ = infer_pose(pose_model, image, mask=segment_mask, rect=rect, bolder=16)

                for keypoints in poses:
                    draw_keypoint(mix, keypoints, labeled=True)

            elif len(faces) >= 2:
                poses, _, _ = infer_pose(pose_model, image, mask=segment_mask, rect=rect, bolder=16)
        
                for keypoint in poses:
                    draw_keypoint(mix, keypoints, labeled=True)
                    instance_mask = infer_instance(instance_model, segment_mask, rect=rect, bolder=16)
                    draw_mask(mix, instance_mask, index2color(k0, 10))
                    k0 = k0 + 1 if k0 < 10 else 0

        imshow(mix, window_name=filepath)
