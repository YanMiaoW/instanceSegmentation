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
from ymlib.dataset_visual import mask2box, draw_keypoint, draw_mask, index2color, draw_box
from ymlib.common import path_decompose, get_minimum_memory_footprint_id, get_git_branch_name

from tsai.face import get_face_detect_model, infer_face_detect
from tsai.segment import get_segment_model, infer_segment
from other.LightPoseEstimation.infer import get_pose_model, infer_pose


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


def get_instance_model(checkpoint_path='/Users/yanmiao/yanmiao/checkpoint/not_exist') -> nn.Module:
    from model.segment import Segment
    instance_model = Segment(3)
    if os.path.exists(checkpoint_path):
        print(f'loading model from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        instance_model.load_state_dict(checkpoint["state_dict"])
    else:
        print('instance_model checkpoint is not found')

    return instance_model


def infer_instance(model, segment_mask, keypoint, image):
    return


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

    face_detect_model = face_detect_model.model.to(device)

    # 遍历预测图片
    print(f'loading image from {args.image_dir}')
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*[jpg,png,jpgerr]")))

    save_dir = os.path.join(args.save_dir, branch_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f'save result mix to {save_dir}')

    for i0, filepath in enumerate(tqdm.tqdm(image_paths)):
        filepath = '/Users/yanmiao/yanmiao/data-common/supervisely/image/05411.png'

        _, basename, _ = path_decompose(filepath)
        result_path = os.path.join(save_dir, f'{basename}.jpg')

        if args.continue_test and os.path.exists(result_path):
            continue

        image = cv.imread(filepath)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mix = image.copy()

        # seg_mask = infer_seg(seg_model, image)

        seg_path = '/Users/yanmiao/yanmiao/data-common/supervisely/segment_mask/05411.png'
        seg_mask = cv.imread(seg_path, cv.IMREAD_GRAYSCALE)

        _, seg_mask_binary = cv.threshold(seg_mask, 127, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(seg_mask_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for j0, (contour, parent) in enumerate(zip(contours, hierarchy[0, :, 3])):
            # 只处理最外层
            if parent != -1:
                continue

            # 获取最外侧外接框
            box = cv.boundingRect(contour)

            faces = infer_face_detect(face_detect_model, image, box)

            # 获取mask
            mask = contour2mask(contours, hierarchy, j0, seg_mask.shape)

            


            # draw_mask(mix, mask, index2color(j0, len(masks)))

        # 
        # if faces is not None:
        #     for j0, box_xyxy in enumerate(faces):
        #         draw_box(mix, box_xyxy, index2color(j0, len(faces)))

        # poses = infer_pose(pose_model, image)
        # for keypoints in poses:
        #     draw_keypoint(mix, keypoints, labeled=True)

        imshow(mix)
        # cv.imwrite(result_path, mix)
