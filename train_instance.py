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

from dataset.common_dataset_api import common_ann_loader, common_aug, common_choice, common_filter, common_transfer, key_combine
from dataset.dataset_visual import mask2box, draw_mask
from common import dict2class, get_git_branch_name, get_minimum_memory_footprint_id, mask_iou
from debug_function import *
from model.segment import Segment

ORDER_PART_NAMES = ["right_shoulder", "right_elbow", "right_wrist",
                    "left_shoulder", "left_elbow", "left_wrist",
                    "right_hip", "right_knee", "right_ankle",
                    "left_hip", "left_knee", "left_ankle",
                    'right_ear', 'right_eye',
                    'left_eye', 'left_ear',
                    'nose', ]

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


def connection2pafs(keypoint, shape, sigma=10):

    pafs = []

    for part1, part2 in CONNECTION_PARTS:

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

            v0 = np.array([x2-x1, y2-y1])
            v0_norm = np.linalg.norm(v0)

            if v0_norm == 0:
                continue

            h, w = shape
            x_min = max(int(round(min(x1, x2) - sigma)), 0)
            x_max = min(int(round(max(x1, x2) + sigma)), w-1)
            y_min = max(int(round(min(y1, y2) - sigma)), 0)
            y_max = min(int(round(max(y1, y2) + sigma)), h-1)

            xs, ys = np.meshgrid(np.arange(x_min, x_max),
                                 np.arange(y_min, y_max))

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

        pafs.append(pafx)
        pafs.append(pafy)

    return pafs


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

        self.heatmap_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.paf_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.results = []

        print('load common dataset from ' + dataset_dir)

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
                              'instance_mask', 'image', 'box', 'body_keypoint'})

                self.results.append(obj)

        self.__getitem__(0)

    def __getitem__(self, index):
        result = self.results[index].copy()

        common_transfer(result)

        image = result[key_combine('image', 'image')]
        box = result[key_combine('box', 'box_xyxy')]

        # 增强

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
        pafs = connection2pafs(keypoint, self.out_size)

        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)
        heatmap_pils = [Image.fromarray(heatmap) for heatmap in heatmaps]
        paf_pils = [Image.fromarray(paf) for paf in pafs]

        image_tensor = self.img_transform(image)
        mask_tensor = self.mask_transform(mask)
        heatmap_tensors = [self.heatmap_transfrom(
            heatmap_pil) for heatmap_pil in heatmaps]
        heatmap_tensor = torch.cat(heatmap_tensors, dim=0)
        paf_pils = [self.paf_transfrom(
            paf_pil) for paf_pil in pafs]
        paf_tensor = torch.cat(paf_pils, dim=0)

        return image_tensor, mask_tensor, heatmap_tensor, paf_tensor

    def __len__(self):
        return len(self.results)


def parse_args():
    args = {
        # "gpu_id": 2,
        "auto_gpu_id": True,
        "continue_train": True,
        "syn_train": True,  # 当多个训练进程共用一个模型存储位置，默认情况会保存最好的模型，如开启syn_train选项，还会将最新模型推送到所有进程。
        "train_dataset_dir": "/data_ssd/ochuman",
        "val_dataset_dir": "/data_ssd/ochuman",
        # "val_dataset_dir": "/data_ssd/hun_sha_di_pian",
        "checkpoint_dir": "/checkpoint/segment",
        # "train_dataset_dir": "/Users/yanmiao/yanmiao/data-common/ochuman",
        # "val_dataset_dir": "/Users/yanmiao/yanmiao/data-common/ochuman",
        # "val_dataset_dir": "/Users/yanmiao/yanmiao/data-common/hun_sha_di_pian",
        # "checkpoint_dir": "/Users/yanmiao/yanmiao/checkpoint/segment",
        # "checkpoint_filename": "union_best.pth",
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

    # 数据导入
    trainset = InstanceCommonDataset(args.train_dataset_dir)

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.cpu_num
    )

    valset = InstanceCommonDataset(args.val_dataset_dir, test=True)

    valloader = DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    # 模型，优化器，损失
    model = Segment(3 + len(CONNECTION_PARTS) * 2 + len(ORDER_PART_NAMES))

    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCELoss()

    # 加载预训练模型
    start_epoch = 0

    iou_max = 0

    branch_name = get_git_branch_name()

    print(f'branch name: {branch_name}')

    if hasattr(args, 'checkpoint_filename'):
        branch_best_path = os.path.join(
            args.checkpoint_dir, args.checkpoint_filename)
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

    # 加载到内存
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

    # 可视化
    show_img_tag = True

    if show_img_tag:
        window_name = f"{branch_name}   {device}    img | label | mix | mask"
        show_img = None

    print("training...")

    epoch = start_epoch

    while epoch < args.epoch:

        loss_total = []
        for i0, (inputs, labels, heatmaps, pafs) in enumerate(trainloader):
            model.train()
            inputs, labels, heatmaps, pafs = inputs.to(
                device), labels.to(device),  heatmaps.to(device), pafs.to(device)
            optimizer.zero_grad()

            outputs = model.train_batch(inputs, heatmaps, pafs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_total.append(loss.item())

            # 打印训练loss
            if i0 % args.show_iter == args.show_iter - 1:
                print(
                    f" [epoch {epoch}]"
                    f" [{i0*args.batch_size}/{len(trainset)}]"
                    f" [loss: {round(sum(loss_total)/len(loss_total),6)}]"
                )
                loss_total = []  # 清空loss

            # 预测
            if i0 % args.val_iter == 0:
                with torch.no_grad():
                    model.eval()

                    # 打印iou
                    def tensors_mean_iou(outputs, labels):
                        ious = []
                        for output, label in zip(outputs, labels):
                            output = output[0].cpu().numpy()*255
                            label = label[0].cpu().numpy()*255
                            ious.append(mask_iou(output, label))
                        return sum(ious)/len(ious)

                    train_batch_iou = tensors_mean_iou(outputs, labels)

                    val_ious = []
                    for j0, (inputs2, labels2, heatmaps2, pafs2) in enumerate(valloader):
                        inputs2, labels2, heatmaps2, pafs2 = inputs2.to(
                            device), labels2.to(device),  heatmaps2.to(device), pafs2.to(device)
                        outputs2 = model.train_batch(inputs2, heatmaps2, pafs2)
                        val_ious.append(tensors_mean_iou(
                            outputs2, labels2))
                        # todo
                        break

                    val_iou = sum(val_ious)/len(val_ious)

                    print(
                        f"{branch_name}",
                        f" [epoch {epoch}]"
                        f" [val_num:{len(valset)}]"
                        f" [train_batch_iou: {round(train_batch_iou,6)}]"
                        f" [val_iou: {round(val_iou,6)}]"
                    )

                    # 可视化
                    if show_img_tag:

                        train_input = inputs[0]
                        train_output = outputs[0]
                        train_label = labels[0]

                        val_input = inputs2[0]
                        val_output = outputs2[0]
                        val_label = labels2[0]

                        def tensor2mask(tensor, thres=0.5):
                            return (tensor[0]*255).cpu().numpy().astype(np.uint8)

                        def tensor2image(tensor):
                            return ((tensor.permute(1, 2, 0)+1)*0.5*255).cpu().numpy().astype(np.uint8)

                        train_img = tensor2image(train_input)
                        train_label_mask = tensor2mask(train_label)
                        train_mask = tensor2mask(train_output)

                        val_img = tensor2image(val_input)
                        val_label_mask = tensor2mask(val_label)
                        val_mask = tensor2mask(val_output)

                        train_mix = train_img.copy()
                        draw_mask(train_mix, train_mask)

                        val_mix = val_img.copy()
                        draw_mask(val_mix, val_mask)

                        train_mask3 = cv.applyColorMap(
                            train_mask, cv.COLORMAP_HOT)
                        val_mask3 = cv.applyColorMap(val_mask, cv.COLORMAP_HOT)

                        # 蓝图变红图
                        # train_mask3 = cv.cvtColor(
                        #     train_mask3, cv.COLOR_BGR2RGB)
                        # val_mask3 = cv.cvtColor(val_mask3, cv.COLOR_BGR2RGB)

                        train_label_mask3 = cv.cvtColor(
                            train_label_mask, cv.COLOR_GRAY2RGB)
                        val_label_mask3 = cv.cvtColor(
                            val_label_mask, cv.COLOR_GRAY2RGB)

                        train_show_img = np.concatenate(
                            [train_img, train_label_mask3, train_mix, train_mask3], axis=1)

                        val_show_img = np.concatenate(
                            [val_img, val_label_mask3, val_mix, val_mask3], axis=1)

                        show_img = np.concatenate(
                            [train_show_img, val_show_img], axis=0)

                        show_img = cv.resize(show_img, (0, 0), fx=0.5, fy=0.5)
                        show_img = cv.cvtColor(show_img, cv.COLOR_RGB2BGR)

                        # TODO 这里重复了，将来封装成统一预测过程

                    # 模型重启
                    if iou_max-val_iou > 0.3:
                        print(
                            f'val_iou too low, reload checkpoint from {branch_best_path}')
                        load_checkpoint(branch_best_path)
                        epoch = start_epoch - 1
                        break

                    # 模型更新
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
                            # TODO 模型保存和读取过程，gpu的加载相关没有做好，容易爆内存， 多个文件同时读写可能冲突

                    # 模型保存
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
