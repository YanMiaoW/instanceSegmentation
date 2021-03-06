import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import random
import math

from imgaug import augmenters as iaa
import imgaug as ia

from ymlib.common_dataset_api import common_ann_loader, common_aug, common_choice, common_filter, common_transfer, key_combine
from ymlib.dataset_visual import mask2box, draw_mask
from ymlib.common import dict2class, get_git_branch_name, get_minimum_memory_footprint_id, get_user_hostname, mean
from ymlib.eval_function import mask_iou
from ymlib.debug_function import *

from model.segment import Segment

ORDER_PART_NAMES = ["right_shoulder", "right_elbow", "right_wrist",
                    "left_shoulder", "left_elbow", "left_wrist",
                    "right_hip", "right_knee", "right_ankle",
                    "left_hip", "left_knee", "left_ankle",
                    'right_ear', 'left_ear',
                    'nose', 'right_eye', 'left_eye']


def keypoint2heatmaps(keypoint, shape, sigma=10, threshold=0.01):

    r = math.sqrt(math.log(threshold)*(-sigma**2))

    heatmaps = []

    for key in ORDER_PART_NAMES:

        heatmap = np.zeros(shape, dtype=np.float32)

        key_type = key_combine(key, 'sub_dict')

        if key_type in keypoint and\
            keypoint[key_type][
                key_combine('status', 'keypoint_status')] == 'vis':

            x, y = keypoint[key_type][key_combine('point', 'point_xy')]
            h, w = shape

            x_min = max(0, int(x - r))
            x_max = min(w-1, int(x+r+1))
            y_min = max(0, int(y - r))
            y_max = min(h-1, int(y+r+1))

            xs = np.arange(x_min, x_max)
            ys = np.arange(y_min, y_max)[:, np.newaxis]

            e_table = np.exp(-((xs - x)**2+(ys - y)**2) / sigma**2)

            idxs = np.where(e_table > threshold)

            heatmap[y_min:y_max, x_min:x_max][idxs] = e_table[idxs]

        heatmaps.append(heatmap)

    return heatmaps


class InstanceCommonDataset(Dataset):

    def __init__(self, dataset_dir, test: bool = False) -> None:
        super().__init__()
        self.test = test

        out_size = (480, 480)
        self.out_size = out_size

        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.mask_transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        self.results = []

        for ann in common_ann_loader(dataset_dir):

            common_choice(ann, key_choices={'image', 'object'})

            objs = ann[key_combine('object', 'sub_list')]
            image_path = ann[key_combine('image', 'image_path')]

            for obj in objs:

                def filter(result):
                    yield 'instance_mask' in result

                    yield 'body_keypoint' in result

                    yield sum(keypoint['status'] != 'missing' for keypoint in result['body_keypoint'].values()) > 9

                    if 'class' in result:
                        yield result['class'] in ['person']

                    yield 'box' in result
                    x0, y0, x1, y1 = result['box']
                    bw, bh = x1-x0, y1-y0
                    yield bw > 50 and bh > 50

                if not common_filter(obj, filter):
                    continue

                obj[key_combine('image', 'image_path')] = image_path

                common_choice(obj, key_choices={
                              'instance_mask', 'image', 'box'})

                self.results.append(obj)

        # self.__getitem__(10)

    def __getitem__(self, index):
        result = self.results[index].copy()

        common_transfer(result)

        image = result[key_combine('image', 'image')]
        box = result[key_combine('box', 'box_xyxy')]

        # ??????

        def sometimes(x): return iaa.Sometimes(0.6, x)

        ih, iw = image.shape[:2]
        x0, y0, x1, y1 = box
        box_center_x = (x0+x1)/2
        box_center_y = (y0+y1)/2
        tx = int(iw / 2 - box_center_x)
        ty = int(ih / 2 - box_center_y)

        if self.test:
            aug = iaa.Affine(translate_px={"x": (tx, tx), "y": (ty, ty)})
        else:
            aug = iaa.Sequential([
                iaa.Affine(translate_px={"x": (tx, tx), "y": (ty, ty)}),
                # sometimes(
                #     iaa.Affine(rotate=(-25, 25)),
                # ),
            ])

        common_aug(result, aug, r=True)

        instance_mask = result[key_combine('instance_mask', 'mask')]
        instance_box = mask2box(instance_mask)

        if instance_box is None:
            instance_box = [0, 0, iw, ih]

        x1, y1, x2, y2 = instance_box
        pad = 16
        left = -x1 + pad
        right = x2 - iw + pad
        top = -y1 + pad
        bottom = y2 - ih + pad
        # aw = int((x2-x1)*0.2)
        # ah = int((y2-y1)*0.2)

        if self.test:
            aug = iaa.Sequential([
                iaa.CropAndPad(((top, top), (right, right),
                                (bottom, bottom), (left, left))),
                iaa.Resize(
                    {"height": self.out_size[0], "width": self.out_size[1]})
            ])
        else:
            aug = iaa.Sequential([
                iaa.CropAndPad(((top, top), (right, right),
                                (bottom, bottom), (left, left))),
                # iaa.CropAndPad(((top-ah, top+ah), (right-aw, right+aw),
                #                 (bottom-ah, bottom+ah), (left-aw, left+aw))),
                # sometimes(iaa.LinearContrast((0.75, 1.5))),
                # sometimes(iaa.AdditiveGaussianNoise(
                #     loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                # sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.2)),
                iaa.Resize(
                    {"height": self.out_size[0], "width": self.out_size[1]})
            ])

        common_aug(result, aug, r=True)

        image = result[key_combine('image', 'image')]
        mask = result[key_combine('instance_mask', 'mask')]
        keypoint = result[key_combine('body_keypoint', 'sub_dict')]

        heatmaps = keypoint2heatmaps(keypoint, self.out_size)

        # image_pil = Image.fromarray(image)
        # mask_pil = Image.fromarray(mask)

        image_tensor = self.img_transform(image)
        mask_tensor = self.mask_transform(mask)

        out = {}
        out['image'] = image
        out['mask'] = mask
        return image_tensor, mask_tensor, out

    def __len__(self):
        return len(self.results)


def collate_fn(batch):
    def deal(samples: list):
        if isinstance(samples[0], torch.Tensor):
            return torch.stack(samples, axis=0)
        else:
            return samples

    return [deal(list(samples)) for samples in zip(*batch)]


def parse_args():

    if get_user_hostname() == YANMIAO_MACPRO_NAME:
        args = {
            # "gpu_id": 2,
            # "auto_gpu_id": True,
            "continue_train": True,
            "syn_train": True,  # ??????????????????????????????????????????????????????????????????????????????????????????????????????syn_train??????????????????????????????????????????????????????
            "train_dataset_dir": "/Users/yanmiao/yanmiao/data-common/ochuman",
            "val_dataset_dir": "/Users/yanmiao/yanmiao/data-common/ochuman",
            # "val_dataset_dir": "/Users/yanmiao/yanmiao/data-common/hun_sha_di_pian",
            "checkpoint_dir": "/Users/yanmiao/yanmiao/checkpoint/segment",
            # "checkpoint_save_path": "",
            # "pretrained_path":"",
            "epoch": 30,
            "show_iter": 20,
            "val_iter": 120,
            "batch_size": 8,
            "cpu_num": 2,
        }

    elif get_user_hostname() == ROOT_201_NAME:
        args = {
            # "gpu_id": 2,
            "auto_gpu_id": True,
            "continue_train": True,
            "syn_train": True,  # ??????????????????????????????????????????????????????????????????????????????????????????????????????syn_train??????????????????????????????????????????????????????
            "train_dataset_dir": "/data_ssd/ochuman",
            "val_dataset_dir": "/data_ssd/ochuman",
            # "val_dataset_dir": "/data_ssd/hun_sha_di_pian",
            "checkpoint_dir": "/checkpoint/segment",
            # "checkpoint_save_path": "",
            # "pretrained_path":"",
            "epoch": 30,
            "show_iter": 20,
            "val_iter": 120,
            "batch_size": 8,
            "cpu_num": 2,
        }

    return dict2class(args)


if __name__ == "__main__":

    args = parse_args()

    # ????????????
    print('load train dataset from ' + args.train_dataset_dir)

    trainset = InstanceCommonDataset(args.train_dataset_dir)

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.cpu_num, collate_fn=collate_fn
    )

    print('load val dataset from ' + args.train_dataset_dir)

    valset = InstanceCommonDataset(args.val_dataset_dir, test=True)

    valloader = DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn
    )

    # ???????????????????????????
    model = Segment(3)

    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCELoss()

    # ?????????????????????
    start_epoch = 0

    iou_max = 0

    branch_name = get_git_branch_name()

    print(f'branch name: {branch_name}')

    if hasattr(args, 'checkpoint_save_path'):
        branch_best_path = args.checkpoint_save_path
    else:
        branch_best_path = os.path.join(
            args.checkpoint_dir, f'{branch_name}_best.pth')

    if os.path.exists(branch_best_path):
        checkpoint = torch.load(branch_best_path)
        iou_max = checkpoint['best']

    def load_checkpoint(checkpoint_path):
        try:
            global start_epoch
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            print('load fail')

    if hasattr(args, 'continue_train') and args.continue_train and os.path.exists(branch_best_path):
        print(f"loading checkpoint from {branch_best_path}")
        load_checkpoint(branch_best_path)

    elif hasattr(args, "pretrained_path") and os.path.exists(args.pretrained_path):
        print(f"pretrained loading checkpoint from {args.pretrained_path}")
        load_checkpoint(args.pretrained_path)
        start_epoch = 0

    # ???????????????
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    if hasattr(args, 'gpu_id'):
        device = torch.device(f"cuda:{args.gpu_id}")
    elif hasattr(args, 'auto_gpu_id') and args.auto_gpu_id:
        device = torch.device(f"cuda:{get_minimum_memory_footprint_id()}")
    else:
        device = 'cpu'

    print(f'device: {device}')

    model = model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # ?????????
    show_img_tag = True

    if show_img_tag:
        window_name = f"{branch_name}   {device}    img | label | mix | mask"
        show_img = None

    print("training...")

    epoch = start_epoch

    while epoch < args.epoch:

        loss_total = []
        for i0, (image_ts, mask_ts, results) in enumerate(trainloader):
            model.train()
            image_ts, mask_ts = image_ts.to(device), mask_ts.to(device)

            optimizer.zero_grad()

            outmask_ts = model.train_batch(image_ts)
            loss = criterion(outmask_ts, mask_ts)
            loss.backward()
            optimizer.step()

            loss_total.append(loss.item())

            # ????????????loss
            if i0 % args.show_iter == args.show_iter - 1:
                print(
                    f" [epoch {epoch}]"
                    f" [{i0*args.batch_size}/{len(trainset)}]"
                    f" [loss: {round(sum(loss_total)/len(loss_total),6)}]"
                )
                loss_total = []  # ??????loss

            # ??????
            if i0 % args.val_iter == 0:
                with torch.no_grad():
                    model.eval()

                    def tensor2mask(tensor):
                        return (tensor[0]*255).cpu().detach().numpy().astype(np.uint8)

                    # ??????iou
                    def tensors_mean_iou(outmask_ts, mask_ts):
                        return mean(mask_iou(tensor2mask(outmask_t), tensor2mask(mask_t)) for outmask_t, mask_t in zip(outmask_ts, mask_ts))

                    train_batch_iou = tensors_mean_iou(outmask_ts, mask_ts)

                    val_ious = []
                    for j0, (vimage_ts, vmask_ts, vresults) in enumerate(valloader):
                        vimage_ts, vmask_ts = vimage_ts.to(
                            device), vmask_ts.to(device)
                        voutmask_ts = model.train_batch(vimage_ts)
                        val_ious.append(tensors_mean_iou(
                            voutmask_ts, vmask_ts))
                        # TODO ????????????????????????
                        break

                    val_iou = mean(val_ious)

                    print(
                        f"{branch_name}",
                        f" {device}",
                        f" [epoch {epoch}]"
                        f" [val_num:{len(valset)}]"
                        f" [train_batch_iou: {round(train_batch_iou,6)}]"
                        f" [val_iou: {round(val_iou,6)}]"
                    )

                    # ?????????
                    if show_img_tag:
                        result = results[0]
                        image = result['image']
                        mask = result['mask']
                        outmask = tensor2mask(outmask_ts[0])

                        vresult = vresults[0]
                        vimage = vresult['image']
                        vmask = vresult['mask']
                        voutmask = tensor2mask(voutmask_ts[0])

                        mix = image.copy()
                        draw_mask(mix, outmask)

                        vmix = vimage.copy()
                        draw_mask(vmix, voutmask)

                        outmask_show = cv.applyColorMap(
                            outmask, cv.COLORMAP_HOT)
                        voutmask_show = cv.applyColorMap(
                            voutmask, cv.COLORMAP_HOT)

                        # ???????????????
                        # train_mask3 = cv.cvtColor(
                        #     train_mask3, cv.COLOR_BGR2RGB)
                        # val_mask3 = cv.cvtColor(val_mask3, cv.COLOR_BGR2RGB)

                        mask3 = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
                        vmask3 = cv.cvtColor(vmask, cv.COLOR_GRAY2RGB)

                        train_show_img = np.concatenate(
                            [image, mask3, mix, outmask_show], axis=1)

                        val_show_img = np.concatenate(
                            [vimage, vmask3, vmix, voutmask_show], axis=1)

                        show_img = np.concatenate(
                            [train_show_img, val_show_img], axis=0)

                        show_img = cv.resize(show_img, (0, 0), fx=0.5, fy=0.5)
                        show_img = cv.cvtColor(show_img, cv.COLOR_RGB2BGR)

                    # ????????????
                    if iou_max-val_iou > 0.3:
                        print(
                            f'val_iou too low, reload checkpoint from {branch_best_path}')
                        load_checkpoint(branch_best_path)
                        epoch = start_epoch - 1
                        break

                    # ????????????
                    if os.path.exists(branch_best_path):
                        checkpoint = torch.load(branch_best_path)
                        if iou_max < checkpoint['best'] or epoch - start_epoch > 10:
                            print(f'update model from {branch_best_path}')
                            iou_max = checkpoint['best']
                            if hasattr(args, 'syn_train') and args.syn_train:
                                print('syn_train...')
                                load_checkpoint(branch_best_path)
                                epoch = start_epoch - 1
                                break

                    # ????????????
                    if val_iou > iou_max and val_iou > 0.7:
                        iou_max = val_iou

                        print("save branch best checkpoint " + branch_best_path)

                        state = {
                            "branch_name": branch_name,
                            "best": iou_max,
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }
                        if not os.path.exists(args.checkpoint_dir):
                            os.makedirs(args.checkpoint_dir)
                        try:
                            torch.save(state, branch_best_path)
                        except:
                            print('save_fail')

            if show_img_tag and show_img is not None:
                cv.imshow(window_name, show_img)
                cv.waitKey(5)

        epoch += 1

    cv.destroyWindow(window_name)
