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
from model.segment import Segment
from tsai.face import get_face_model, infer_face
from tsai.segment import get_segment_model, infer_seg
from other.LightPoseEstimation.infer import get_pose_estimation_model, pose_infer


def parse_args():
    parser = argparse.ArgumentParser(description='demo image')
    parser.add_argument('-i', '--image-dir',
                        help='image test dir', required=True)
    parser.add_argument('-o', '--save-dir',
                        help='output save dir', required=True)
    parser.add_argument('--checkpoint-dir',
                        help='checkpoint save dir')
    parser.add_argument('--checkpoint-path',
                        help='checkpoint save path')
    parser.add_argument('--continue-test', action='store_true',
                        help='skip existing file.')
    parser.add_argument('--auto-gpu-id', action='store_true',
                        help='gpu inference')

    args = parser.parse_args()
    return args


def infer(model, image, keypoint):
    return image


if __name__ == "__main__":
    args = parse_args()

    branch_name = get_git_branch_name()

    save_dir = os.path.join(args.save_dir, branch_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 加载模型
    if args.checkpoint_path is not None:
        branch_best_path = args.checkpoint_path
    elif args.checkpoint_dir is not None:
        branch_best_path = os.path.join(
            args.checkpoint_dir, f'{branch_name}_best.pth')
    else:
        branch_best_path = ''

    instance_model = Segment(3)
    if os.path.exists(branch_best_path):
        print(f'loading model from {branch_best_path}')
        checkpoint = torch.load(branch_best_path)
        instance_model.load_state_dict(checkpoint["state_dict"])

    pose_model = get_pose_estimation_model()

    # 加载到内存
    if args.auto_gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        device = torch.device(f"cuda:{get_minimum_memory_footprint_id()}")
    else:
        device = 'cpu'

    instance_model = instance_model.to(device)
    pose_model = pose_model.to(device)

    seg_model = get_segment_model()
    face_model = get_face_model(device)

    print(f'loading image from {args.image_dir}')
    image_paths = sorted(glob.glob(os.path.join(
        args.image_dir, "*[jpg,png,jpgerr]")))

    print(f'save result mix to {args.save_dir}')

    for i0, filepath in enumerate(tqdm.tqdm(image_paths)):
        _, basename, _ = path_decompose(filepath)
        result_path = os.path.join(save_dir, f'{basename}.jpg')

        if args.continue_test:
            if os.path.exists(result_path):
                continue

        image = cv.imread(filepath)
        mix = image.copy()

        seg_mask = infer_seg(seg_model, image)
        draw_mask(mix, seg_mask)

        # faces = infer_face(face_model, image)
        # if faces is not None:
        #     for j0, xyxy in enumerate(faces):
        #         draw_box(mix, xyxy, index2color(j0, len(faces)))

        # poses = pose_infer(pose_model, image)
        # for keypoints in poses:
        #     draw_keypoint(mix, keypoints, labeled=True)

        cv.imwrite(result_path, mix)
