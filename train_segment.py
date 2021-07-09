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

from imgaug import augmenters as iaa
import imgaug as ia
from pygit2 import Repository

from dataset.common_dataset_api import *
from model.segment import Segment
from dataset.common_dataset_api import *
from common_func import path_decompose
from debug_function import *


class SegmentCommonDataset(Dataset):

    def __init__(self, dataset_dir, test: bool = False) -> None:
        super().__init__()

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

        for result in common_ann_loader(dataset_dir):

            def ann_filter(result):

                yield all(key in result for key in ['image', 'segment_mask'])

                if 'meta' in result:
                    meta = result['meta']
                    yield meta['width'] * meta['height'] < 1200*1200

                if 'class' in result:
                    yield result['class'] in ['person']

            common_choice(result, key_choices={
                          'image', 'segment_mask', 'meta', 'class'})

            if not common_filter(result, ann_filter):
                continue

            self.results.append(result)

    def __getitem__(self, index):
        result = self.results[index].copy()

        common_transfer(result)
        common_aug(result, self.aug)

        image = result[key_combine('image', 'image')]
        mask = result[key_combine('segment_mask', 'mask')]

        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)

        image_tensor = self.transform(image_pil)
        mask_tensor = self.label_transform(mask_pil)

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.results)


def parse_args():
    args = {
        "gpu_id": 2,
        "epoch_model": 0,
        "continue_train": False,
        "train_dataset_dir": "/data_ssd/supervisely",
        "val_dataset_dir": "/data_ssd/val",
        "checkpoint_dir": "/checkpoint/segment",
        # "pretrained_path": "/checkpoint/segmentation_20200618/199.pth",
        "epoch": 30,
        "show_iter": 20,
        "val_iter": 120,
        "batch_size": 8,
        "cpu_num": 4,
    }

    class Dict2Class(object):
        def __init__(self, args):
            for key in args:
                setattr(self, key, args[key])

    args = Dict2Class(args)
    return args


if __name__ == "__main__":
    repo = Repository('.')
    head = repo.lookup_reference('HEAD').resolve()
    head = repo.head
    branch_name = head.name.split('/')[-1]

    args = parse_args()
    show_img_tag = True

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    trainset = SegmentCommonDataset(args.train_dataset_dir)

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.cpu_num
    )

    valset = SegmentCommonDataset(args.val_dataset_dir, test=True)

    valloader = DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    model = Segment(1)

    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCELoss()

    start_epoch = 0
    if args.continue_train:
        checkpoint_path = os.path.join(
            args.checkpoint_dir, f"{args.epoch_model}.pth")
        if os.path.exists(checkpoint_path):
            print(f"loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    elif hasattr(args, "pretrained_path") and os.path.exists(args.pretrained_path):
        checkpoint_path = args.pretrained_path
        print(f"pretrained loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    iou_max = 0

    iou_max_best = 0

    branch_best_path = os.path.join(
        args.checkpoint_dir, f'{branch_name}_best.pth')

    best_path = os.path.join(args.checkpoint_dir, f'best.pth')

    if os.path.exists(branch_best_path):

        branch_best_checkpoint = torch.load(os.path.join(
            args.checkpoint_dir, branch_best_path))

        iou_max = branch_best_checkpoint['best']

    if os.path.exists(best_path):

        best_checkpoint = torch.load(os.path.join(
            args.checkpoint_dir, best_path))

        iou_max_best = best_checkpoint['best']

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if show_img_tag:
        window_name = f"{branch_name} img | mix | mask"
        show_img = None

    print("training...")
    for epoch in range(args.epoch):

        if args.continue_train and epoch < start_epoch:
            continue

        loss_total = 0.0
        for i0, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
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

                train_output = outputs[:1]
                train_img_tensor = inputs[:1]

                train_iou = mask_iou(outputs, labels)
                with torch.no_grad():
                    total_iou = 0

                    for j0, (inputs2, labels2) in enumerate(valloader):
                        inputs2, labels2 = inputs2.to(
                            device), labels2.to(device)
                        outputs, _ = model(inputs2)
                        outputs = torch.sigmoid(outputs)

                        iou = mask_iou(outputs, labels2)
                        total_iou += iou
                    iou = total_iou / len(valloader)

                    print(
                        f"{branch_name}",
                        f" [epoch {epoch}]"
                        f" [val_num:{len(valset)}]"
                        f" [train_iou: {round(train_iou,6)}]"
                        f" [val_iou: {round(iou,6)}]"
                    )

                    def get_mix(output, img_tensor: torch.Tensor):
                        output = (
                            (output * 255)
                            .cpu()
                            .permute(0, 2, 3, 1)
                            .numpy()[0]
                            .astype(np.uint8)
                        )
                        output = np.repeat(output, 3, axis=2)

                        img = ((img_tensor + 1)*0.5*255).cpu()\
                            .permute(0, 2, 3, 1) .numpy()[0].astype(np.uint8)
                        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

                        mix = img.copy()
                        select = (output > 127).max(2)
                        mix[select] = (
                            np.array([0, 255, 255], dtype=np.uint8) // 2 +
                            mix[select] // 2
                        )
                        return mix, output, img

                    img_tensor, mask = random.choice(valset)
                    output, _ = model(img_tensor[np.newaxis, :].to(device))
                    output = torch.sigmoid(output)

                    mix, output, img = get_mix(
                        output, img_tensor[np.newaxis, :])

                    train_mix, train_output, train_img = get_mix(
                        train_output, train_img_tensor)

                    train_show_img = np.concatenate(
                        [train_img, train_mix, train_output], axis=1)

                    val_show_img = np.concatenate([img, mix, output], axis=1)

                    show_img2 = np.concatenate(
                        [train_show_img, val_show_img], axis=0)

                    show_img = cv.resize(show_img2, (0, 0), fx=0.5, fy=0.5)

                    if iou > iou_max and iou > 0.7:
                        iou_max = iou

                        print("save branch best checkpoint " + branch_best_path)

                        state = {
                            "branch_name": branch_name,
                            "best": iou_max,
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }
                        torch.save(state, branch_best_path)

                    if iou_max > iou_max_best:
                        iou_max_best = iou_max
                        print("save best checkpoint " + best_path)

                        state = {
                            "branch_name": branch_name,
                            "best": iou_max,
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }
                        torch.save(state, branch_best_path)

                model.train()

            if show_img_tag and show_img is not None:
                cv.imshow(window_name, show_img)
                cv.waitKey(5)

        # print(f"save checkpoint {epoch}.pth")
        # state = {
        #     "epoch": epoch + 1,
        #     "state_dict": model.state_dict(),
        #     "optimizer": optimizer.state_dict(),
        # }
        # torch.save(
        #     state,
        #     os.path.join(args.checkpoint_dir, f"{epoch}.pth"),
        # )

    cv.destroyWindow(window_name)