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

from imgaug import augmenters as iaa
import imgaug as ia

from dataset.common_dataset_api import common_ann_loader, common_aug, common_choice, common_filter, common_transfer, key_combine
from dataset.dataset_visual import mask2box, draw_mask
from common import dict2class, get_git_branch_name, get_minimum_memory_footprint_id, mask_iou
from debug_function import *
from model.segment import Segment


class InstanceCommonDataset(Dataset):

    def __init__(self, dataset_dir, test: bool = False) -> None:
        super().__init__()
        self.test = test

        out_size = (480, 480)

        self.transform = transforms.Compose(
            [
                transforms.Resize(out_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.label_transform = transforms.Compose(
            [transforms.Resize(out_size), transforms.ToTensor()]
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

                    if 'class' in result:
                        yield result['class'] in ['person']

                    yield 'box' in result
                    x0, y0, x1, y1 = result['box']
                    bw, bh = x1-x0, y1-y0
                    yield bw > 50 and bh > 50

                if not common_filter(obj, filter):
                    continue

                obj[key_combine('instance_image', 'image_path')] = image_path

                common_choice(obj, key_choices={
                              'instance_mask', 'instance_image', 'box'})

                self.results.append(obj)

    def __getitem__(self, index):
        result = self.results[index].copy()

        common_transfer(result)

        image = result[key_combine('instance_image', 'image')]
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
                sometimes(
                    iaa.Affine(rotate=(-25, 25)),
                ),
            ])

        # common_aug(result, aug, r=True)

        instance_mask = result[key_combine('instance_mask', 'mask')]
        instance_box = mask2box(instance_mask)

        if instance_box is None:
            instance_box = [0, 0, iw, ih]

        x1, y1, x2, y2 = instance_box
        left = -x1
        right = x2 - iw
        top = -y1
        bottom = y2 - ih
        aw = int((x2-x1)*0.2)
        ah = int((y2-y1)*0.2)

        if self.test:
            aug = iaa.CropAndPad(((top, top), (right, right),
                                  (bottom, bottom), (left, left)))
        else:
            aug = iaa.Sequential([
                iaa.CropAndPad(((top, top), (right, right),
                                (bottom, bottom), (left, left))),
                # iaa.CropAndPad(((top-ah, top+ah), (right-aw, right+aw),
                #                 (bottom-ah, bottom+ah), (left-aw, left+aw))),
                iaa.Fliplr(0.5),
                # sometimes(iaa.LinearContrast((0.75, 1.5))),
                # sometimes(iaa.AdditiveGaussianNoise(
                #     loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                # sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.2)),
            ])

        # common_aug(result, aug, r=True)

        image = result[key_combine('instance_image', 'image')]
        mask = result[key_combine('instance_mask', 'mask')]

        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)

        image_tensor = self.transform(image_pil)
        mask_tensor = self.label_transform(mask_pil)

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.results)


def parse_args():
    args = {
        # "gpu_id": 2,
        "auto_gpu_id": True,
        # "continue_train": True,
        "syn_train": False,  # 当多个训练进程共用一个模型存储位置，默认情况会保存最好的模型，如开启syn_train选项，还会将最新模型推送到所有进程。
        "train_dataset_dir": "/data_ssd/ochuman",
        "val_dataset_dir": "/data_ssd/hun_sha_di_pian",
        "checkpoint_dir": "/checkpoint/segment",
        # "train_dataset_dir": "/Users/yanmiao/yanmiao/data-common/ochuman",
        # "val_dataset_dir": "/Users/yanmiao/yanmiao/data-common/hun_sha_di_pian",
        # "checkpoint_dir": "/Users/yanmiao/yanmiao/checkpoint/segment",
        # "checkpoint_filename": "union_best.pth",
        # "pretrained_path":"",
        "epoch": 30,
        "show_iter": 20,
        "val_iter": 120,
        "batch_size": 8,
        "cpu_num": 4,
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
    model = Segment()

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
        global start_epoch
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
            

    if args.continue_train and os.path.exists(branch_best_path):
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
        for i0, (inputs, labels) in enumerate(trainloader):
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model.train_batch(inputs)
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
                    for j0, (inputs2, labels2) in enumerate(valloader):
                        inputs2, labels2 = inputs2.to(
                            device), labels2.to(device)
                        outputs2 = model.train_batch(inputs2)
                        val_ious.append(tensors_mean_iou(outputs2, labels2))

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

                        train_mask3 = cv.cvtColor(
                            train_mask, cv.COLOR_GRAY2RGB)
                        train_label_mask3 = cv.cvtColor(
                            train_label_mask, cv.COLOR_GRAY2RGB)
                        val_mask3 = cv.cvtColor(val_mask, cv.COLOR_GRAY2RGB)
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
                        if iou_max < checkpoint['best']:
                            iou_max = checkpoint['best']
                            if hasattr(args, 'syn_train') and args.syn_train:
                                print('syn_train...')
                                load_checkpoint(branch_best_path)
                                epoch = start_epoch - 1
                                break  

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

                        torch.save(state, branch_best_path)
                            

            if show_img_tag and show_img is not None:
                cv.imshow(window_name, show_img)
                cv.waitKey(5)

        epoch += 1

    cv.destroyWindow(window_name)
