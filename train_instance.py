import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.optim import Adam

from imgaug import augmenters as iaa
import imgaug as ia

from ymlib.common_dataset_api import common_ann_loader, common_aug, common_choice, common_filter, common_transfer, key_combine
from ymlib.dataset_visual import draw_mask
from ymlib.dataset_util import mask_iou, mask2box, get_downsample_ratio, hw2xyxy, xyxy2ltrb, xyxy2center, xyxy2hw
from ymlib.common import dict2class, get_git_branch_name, get_maximum_free_memory_gpu_id, get_user_hostname, mean
from ymlib.debug_function import *

from model.segment import Segment
from instanceSegmentation.infer import ORDER_PART_NAMES, CONNECTION_PARTS, MODEL_INPUT_SIZE, keypoint2heatmaps, connection2pafs


class InstanceCommonDataset(Dataset):
    def __init__(self, dataset_dir, test: bool = False) -> None:
        super().__init__()

        self.test = test

        self.results = []

        for ann in common_ann_loader(dataset_dir):

            common_choice(ann, key_choices={'image', 'object', 'segment_mask'})

            objs = ann[key_combine('object', 'sub_list')]
            image_path = ann[key_combine('image', 'image_path')]
            segment_path = ann[key_combine('segment_mask', 'mask_path')]

            for obj in objs:

                def filter(result):
                    yield 'instance_mask' in result

                    yield 'body_keypoints' in result

                    yield sum(keypoint['status'] != 'missing' for keypoint in result['body_keypoints'].values()) > 9

                    if 'class' in result:
                        yield result['class'] in ['person']

                    yield 'box' in result
                    x0, y0, x1, y1 = result['box']
                    bw, bh = x1 - x0, y1 - y0
                    yield bw > 50 and bh > 50

                if common_filter(obj, filter):

                    obj[key_combine('image', 'image_path')] = image_path
                    obj[key_combine('segment_mask', 'mask_path')] = segment_path

                    common_choice(obj, key_choices={'instance_mask', 'image', 'box', 'body_keypoints', 'segment_mask'})

                    self.results.append(obj)

        self.__getitem__(201)

    def __getitem__(self, index):
        result = self.results[index].copy()

        common_transfer(result)

        # 增强
        def sometimes(x):
            return iaa.Sometimes(0.6, x)

        box_type = key_combine('box', 'box_xyxy')
        segment_mask_type = key_combine('segment_mask', 'mask')

        # 降分辨率
        common_aug(result, iaa.Resize(get_downsample_ratio(xyxy2hw(result[box_type]), MODEL_INPUT_SIZE, base_on='short')), r=True)

        # 移到中心并旋转
        cx, cy = xyxy2center(hw2xyxy(result[segment_mask_type].shape))
        box_cx, box_cy = xyxy2center(result[box_type])
        common_aug(result,
                   iaa.Affine(
                       translate_px={
                           'x': int(cx - box_cx),
                           'y': int(cy - box_cy)
                       },
                       rotate=(-15, 15) if not self.test else None,
                   ),
                   r=True)

        # 裁剪
        xyxy = mask2box(result[segment_mask_type])
        left, top, right, bottom = xyxy2ltrb(xyxy, result[segment_mask_type].shape)
        left, top, right, bottom = left + 32, top + 32, right + 32, bottom + 32
        bh, bw = xyxy2hw(xyxy)
        ah, aw = int(bh * 0.1), int(bw * 0.1)
        common_aug(result,
                   iaa.CropAndPad(px=((top - ah, top + ah), (right - aw, right + aw), (bottom - ah, bottom + ah),
                                      (left - aw, left + aw)) if not self.test else (top, right, bottom, left)),
                   r=True)

        # 增强
        common_aug(
            result,
            iaa.Sequential([
                sometimes(iaa.LinearContrast((0.75, 1.5))),
                sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * .99), per_channel=0.5)),
                sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.2)),
            ] if not self.test else None), r=True)

        common_aug(result, iaa.Resize({"height": MODEL_INPUT_SIZE[0], "width": MODEL_INPUT_SIZE[1]}), r=True)

        image = result[key_combine('image', 'image')]
        segment_mask = result[key_combine('segment_mask', 'mask')]
        instance_mask = result[key_combine('instance_mask', 'mask')]

        body_keypoints = result[key_combine('body_keypoints', 'sub_dict')]
        pafs, paf_show = connection2pafs(body_keypoints, MODEL_INPUT_SIZE)
        heatmaps, heatmap_show = keypoint2heatmaps(body_keypoints, MODEL_INPUT_SIZE)

        to_tensor = transforms.ToTensor()
        normal = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        image_tensor = normal(to_tensor(image))
        segment_mask_tensor = to_tensor(segment_mask)
        instance_mask_tensor = to_tensor(instance_mask)
        heatmaps_tensor = to_tensor(heatmaps)
        pafs_tensor = to_tensor(pafs)

        input_tensor = torch.cat([image_tensor, segment_mask_tensor, heatmaps_tensor, pafs_tensor], dim=0)

        out = {}
        out['image'] = image
        out['instance_mask'] = instance_mask
        out['heatmapShow'] = heatmap_show
        out['pafShow'] = paf_show

        return input_tensor, instance_mask_tensor, out

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
            "syn_train": True,  # 当多个训练进程共用一个模型存储位置，默认情况会保存最好的模型，如开启syn_train选项，还会将最新模型推送到所有进程。
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
            "cpu_num": 0,
        }

    elif get_user_hostname() == ROOT_201_NAME:
        args = {
            # "gpu_id": 2,
            "auto_gpu_id": True,
            "continue_train": True,
            "syn_train": True,  # 当多个训练进程共用一个模型存储位置，默认情况会保存最好的模型，如开启syn_train选项，还会将最新模型推送到所有进程。
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

    # 数据导入
    print('load train dataset from ' + args.train_dataset_dir)
    trainset = InstanceCommonDataset(args.train_dataset_dir)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.cpu_num, collate_fn=collate_fn)

    print('load val dataset from ' + args.train_dataset_dir)
    valset = InstanceCommonDataset(args.val_dataset_dir, test=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 模型，优化器，损失
    model = Segment(3 + 1 + len(CONNECTION_PARTS) * 2 + len(ORDER_PART_NAMES))

    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = Adam(model.parameters())

    criterion = nn.BCELoss()

    # 加载预训练模型
    branch_name = get_git_branch_name()
    print(f'branch name: {branch_name}')

    # 显存设备选择
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    if hasattr(args, 'gpu_id'):
        device = torch.device(f"cuda:{args.gpu_id}")
    elif hasattr(args, 'auto_gpu_id') and args.auto_gpu_id:
        device = torch.device(f"cuda:{get_maximum_free_memory_gpu_id()}")
    else:
        device = 'cpu'

    print(f'device: {device}')

    # 加载
    start_epoch = 0
    iou_max = 0

    if hasattr(args, 'checkpoint_save_path'):
        branch_best_path = args.checkpoint_save_path
    else:
        branch_best_path = os.path.join(args.checkpoint_dir, f'{branch_name}_best.pth')

    def load_checkpoint(checkpoint_path):
        try:
            global start_epoch, device, model, optimizer, iou_max
            model = model.cpu()
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
            start_epoch = checkpoint["epoch"]
            iou_max = max(iou_max, checkpoint['best'])
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

    # 可视化
    show_img_tag = True

    if show_img_tag:
        window_name = f"{branch_name}   {device}    img | label | heatmap | paf | mix | mask"
        show_img = None

    print("training...")

    epoch = start_epoch

    while epoch < args.epoch:

        loss_total = []
        for i0, (input_ts, instance_mask_ts, outs) in enumerate(trainloader):
            model.train()
            input_ts, instance_mask_ts = input_ts.to(device), instance_mask_ts.to(device)

            optimizer.zero_grad()

            output_ts = torch.sigmoid(model(input_ts))

            loss = criterion(output_ts, instance_mask_ts)
            loss.backward()
            optimizer.step()

            loss_total.append(loss.item())

            # 打印训练loss
            if i0 % args.show_iter == args.show_iter - 1:
                print(f" [epoch {epoch}]"
                      f" [{i0*args.batch_size}/{len(trainset)}]"
                      f" [loss: {round(sum(loss_total)/len(loss_total),6)}]")
                loss_total = []  # 清空loss

            # 预测
            if i0 % args.val_iter == 0:
                with torch.no_grad():
                    model.eval()

                    def tensor2mask(tensor):
                        return (tensor[0] * 255.99).cpu().detach().numpy().astype(np.uint8)

                    def tensors_mean_iou(outmask_ts, mask_ts):
                        return mean(
                            mask_iou(tensor2mask(outmask_t), tensor2mask(mask_t))
                            for outmask_t, mask_t in zip(outmask_ts, mask_ts))

                    # 打印iou
                    train_batch_iou = tensors_mean_iou(output_ts, instance_mask_ts)

                    val_ious = []
                    for j0, (vinput_ts, vinstance_mask_ts, vouts) in enumerate(valloader):
                        vinput_ts, vinstance_mask_ts = vinput_ts.to(device), vinstance_mask_ts.to(device)

                        voutput_ts = torch.sigmoid(model(vinput_ts))

                        val_ious.append(tensors_mean_iou(voutput_ts, vinstance_mask_ts))
                        if len(val_ious) > 30:
                            break

                    val_iou = mean(val_ious)

                    print(
                        f"{branch_name}", f" {device}", f" [epoch {epoch}]"
                        f" [val_num:{len(valset)}]"
                        f" [train_batch_iou: {round(train_batch_iou,6)}]"
                        f" [val_iou: {round(val_iou,6)}]")

                    # 可视化
                    if show_img_tag:
                        result = outs[0]
                        image = result['image']
                        instance_mask = result['instance_mask']
                        heatmap_show = result['heatmapShow']
                        paf_show = result['pafShow']
                        outmask = tensor2mask(output_ts[0])

                        vresult = vouts[0]
                        vimage = vresult['image']
                        vinstance_mask = vresult['instance_mask']
                        vheatmap_show = vresult['heatmapShow']
                        vpaf_show = vresult['pafShow']
                        voutmask = tensor2mask(voutput_ts[0])

                        mix = image.copy()
                        draw_mask(mix, outmask)

                        vmix = vimage.copy()
                        draw_mask(vmix, voutmask)

                        outmask_show = cv.applyColorMap(outmask, cv.COLORMAP_HOT)
                        voutmask_show = cv.applyColorMap(voutmask, cv.COLORMAP_HOT)

                        # 蓝图变红图
                        # train_mask3 = cv.cvtColor(
                        #     train_mask3, cv.COLOR_BGR2RGB)
                        # val_mask3 = cv.cvtColor(val_mask3, cv.COLOR_BGR2RGB)

                        instance_mask3 = cv.cvtColor(instance_mask, cv.COLOR_GRAY2BGR)
                        vinstance_mask3 = cv.cvtColor(vinstance_mask, cv.COLOR_GRAY2BGR)

                        train_show_img = np.concatenate([image, instance_mask3, heatmap_show, paf_show, mix, outmask_show], axis=1)

                        val_show_img = np.concatenate([vimage, vinstance_mask3, vheatmap_show, vpaf_show, vmix, voutmask_show], axis=1)

                        show_img = np.concatenate([train_show_img, val_show_img], axis=0)

                        show_img = cv.resize(show_img, (0, 0), fx=0.5, fy=0.5)

                    # 模型退化重启
                    if iou_max - val_iou > 0.3:
                        print(f'val_iou too low, reload checkpoint from {branch_best_path}')
                        load_checkpoint(branch_best_path)
                        epoch = start_epoch - 1
                        break

                    # 模型更新
                    if os.path.exists(branch_best_path) and hasattr(args, 'syn_train') and args.syn_train:
                        checkpoint = torch.load(branch_best_path, map_location=torch.device('cpu'))
                        if iou_max < checkpoint['best']:
                            print(f'syn_train update model from {branch_best_path}')
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
                        try:
                            torch.save(state, branch_best_path)
                        except:
                            print('save_fail')

            if show_img_tag and show_img is not None:
                cv.imshow(window_name, show_img)
                cv.waitKey(5)

        epoch += 1

    cv.destroyWindow(window_name)
