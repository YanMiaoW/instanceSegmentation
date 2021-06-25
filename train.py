import torch
import torch.nn as nn
import torch.nn.functional as F
from debug_function import *
import numpy as np
import os
import glob
import tqdm
import json
import cv2 as cv
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from model.segment import Segment
import random


class SegmentDataset(Dataset):
    def __init__(self, dataset_dir, transform, label_transform):
        mask_dir = os.path.join(dataset_dir, "mask")
        image_dir = os.path.join(dataset_dir, "image")
        img_paths = []
        mask_paths = []
        for img_path in glob.glob(os.path.join(image_dir, "*")):
            dirname, basename, ext = path_decompose(img_path)
            mask_path = os.path.join(mask_dir, f"{basename}.{ext}")
            if not os.path.exists(mask_path):
                continue
            img_paths.append(img_path)
            mask_paths.append(mask_path)

        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        img = self.transform(img)

        mask = self.label_transform(mask)

        return img, mask, img_path

    def __len__(self):
        return len(self.img_paths)


def mask_iou(masks, labels):
    iou_total = 0
    for mask, label in zip(masks, labels):
        mask, label = mask[0] > 0.5, label[0] > 0.5
        union = mask | label
        inter = mask & label
        iou = inter.sum() / union.sum()
        iou_total += iou.item()

    return iou_total / len(masks)
    # return iou


def path_decompose(path):
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)
    ext = os.path.splitext(path)[-1][1:]
    basename = os.path.splitext(basename)[0]
    return dirname, basename, ext


def parse_args():
    args = {
        "gpu_id": 0,
        "epoch_model": 0,
        "continue_train": False,
        # "train_dataset_dir": "/data_ssd/supervislyHumanSegmentation",
        "train_dataset_dir": "/data_ssd/OCHumanSegmentation",
        "val_dataset_dir": "/data_ssd/valSegmentation",
        "checkpoint_dir": "/checkpoint/segmentation_20200621",
        # "pretrained_path": "/checkpoint/segmentation_20200618/199.pth",
        "epoch": 100,
        "show_iter": 20,
        "val_iter": 120,
        "batch_size": 16,
        "cpu_num": 8,
    }

    class Dict2Class(object):
        def __init__(self, args):
            for key in args:
                setattr(self, key, args[key])

    args = Dict2Class(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    show_img_tag = True

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    transform = transforms.Compose(
        [
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    label_transform = transforms.Compose(
        [transforms.Resize((480, 480)), transforms.ToTensor()]
    )

    trainset = SegmentDataset(args.train_dataset_dir, transform, label_transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.cpu_num
    )

    valset = SegmentDataset(args.val_dataset_dir, transform, label_transform)

    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    model = Segment(1)
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

    start_epoch = 0
    if args.continue_train:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.epoch_model}.pth")
        if os.path.exists(checkpoint_path):
            print(f"loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    elif hasattr(args,"pretrained_path") and os.path.exists(args.pretrained_path):
        checkpoint_path = args.pretrained_path
        print(f"pretrained loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    criterion = nn.BCELoss()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if show_img_tag:
        window_name = "img | mix | mask"
        show_img = None

    iou_max = 0
    print("training...")
    for epoch in range(args.epoch):

        if args.continue_train and epoch < start_epoch:
            continue

        loss_total = 0.0
        for i0, (inputs, labels, _) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, _ = model(inputs)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            if i0 % args.show_iter == args.show_iter - 1:
                print(
                    f" [epoch {epoch}]"
                    f" [{i0*args.batch_size}/{len(trainset)}]"
                    f" [loss: {round(loss_total/(args.show_iter*args.batch_size),6)}]"
                )
                loss_total = 0

            if i0 % args.val_iter == 0:
                model.eval()

                train_iou = mask_iou(outputs, labels)
                with torch.no_grad():
                    total_iou = 0

                    for j0, (inputs2, labels2, _) in enumerate(valloader):
                        inputs2, labels2 = inputs2.to(device), labels2.to(device)
                        outputs, _ = model(inputs2)
                        outputs = torch.sigmoid(outputs)

                        iou = mask_iou(outputs, labels2)
                        total_iou += iou
                    iou = total_iou / len(valloader)

                    print(
                        f" [epoch {epoch}]"
                        f" [val_num:{len(valset)}]"
                        f" [train_iou: {round(train_iou,6)}]"
                        f" [val_iou: {round(iou,6)}]"
                    )

                    img_tensor, mask, img_path = random.choice(valset)
                    output, _ = model(img_tensor[np.newaxis, :].to(device))
                    output = torch.sigmoid(output)
                    # mask = mask.permute(1,2,0).numpy()
                    # mask = np.repeat(output,3,axis =2)
                    img = cv.imread(img_path)
                    img = cv.resize(img, (480, 480))
                    output = (
                        (output * 255)
                        .cpu()
                        .permute(0, 2, 3, 1)
                        .numpy()[0]
                        .astype(np.uint8)
                    )
                    # output = np.ones(output.shape, dtype=np.uint8) * 255
                    output = np.repeat(output, 3, axis=2)
                    mix = img.copy()
                    select = (output > 127).max(2)
                    mix[select] = (
                        np.array([0, 255, 255], dtype=np.uint8) // 2 + mix[select] // 2
                    )
                    show_img = np.concatenate([img, mix, output], axis=1)

                    if iou > iou_max and iou > 0.7:
                        iou_max = iou
                        print(
                            "save best checkpoint "
                            + f"best_epoch_{epoch}_iou_{int(iou*100)}.pth"
                        )

                        state = {
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }
                        torch.save(
                            state,
                            os.path.join(
                                args.checkpoint_dir,
                                f"best_epoch_{epoch}_iou_{int(iou*100)}.pth",
                            ),
                        )

                model.train()

            if show_img_tag and show_img is not None:
                cv.imshow(window_name, show_img)
                cv.waitKey(5)
                    

        print(f"save checkpoint {epoch}.pth")
        # torch.save(
        #     model.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}.pth")
        # )

        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(
            state,
            os.path.join(args.checkpoint_dir, f"{epoch}.pth"),
        )
        
    cv.destroyWindow(window_name)
    
