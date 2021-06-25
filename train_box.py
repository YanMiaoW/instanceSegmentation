import torch
import torch.nn as nn
from debug_function import *
import numpy as np
import os
import glob
import tqdm
import cv2 as cv
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from model.detect import Detect
import random
from itertools import product


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.confidence_loss = nn.BCELoss()
        self.coordinate_loss = nn.SmoothL1Loss()

    def forward(self, x, label):
        x, label = x.to("cpu"), label.to("cpu")
        c = x.shape[-1]
        x, label = torch.reshape(x, (-1, c)), torch.reshape(label, (-1, c))

        loss = 0

        select = label[:, 0] > 0.5
        sx = x[~select]
        slabel = label[~select]

        c = (sx[:, 0] - slabel[:, 0]).abs()
        c = c.argsort(descending=True)
        c = c[: select.sum() * 3]
        not_select = c

        x[:, 0] = torch.sigmoid(x[:, 0])
        loss += self.confidence_loss(x[:,0][select],label[:,0][select])
        loss += self.confidence_loss(x[:,0][not_select],label[:,0][not_select])
        # loss += self.confidence_loss(x[:, 0], label[:, 0])
        # select = label[:, 0] > 0.5
        # loss += self.class_loss(x[select][:, 0], label[select][:, 0])
        # loss += self.class_loss(x[not_select][:, 0], label[not_select][:, 0])
        # select = label[:, :, :, 0] > 0
        # loss += self.mse_loss(x[select][:, 1:], label[select][:, 1:])
        return loss


def box_eval(x, label):
    x, label = x.to("cpu"), label.to("cpu")
    c = x.shape[-1]
    x = torch.reshape(x, (-1, c))
    label = torch.reshape(label, (-1, c))

    select = label[:, 0] > 0.5
    x[:, 0] = torch.sigmoid(x[:, 0])

    a = x[:, 0][select] > 0.5
    class_acc_fore = sum(a) / len(a)
    
    b = x[:,0][~select] < 0.5
    class_acc_back= sum(b) / len(b)

    [ax, ay, aw, ah] = (x[:, 1:] - label[:, 1:]).abs().mean(0)
    class_acc_fore = class_acc_fore.item() if isinstance(class_acc_fore, torch.Tensor) else class_acc_fore
    class_acc_back = class_acc_back.item() if isinstance(class_acc_back, torch.Tensor) else class_acc_back

    ax, ay, aw, ah = ax.item(), ay.item(), aw.item(), ah.item()
    return class_acc_fore,class_acc_back, ax, ay, aw, ah


class DetectionDataset(Dataset):
    def __init__(self, dataset_dir, size):
        img_paths = []
        data_paths = []
        for img_path in glob.glob(os.path.join(dataset_dir, "image", "*")):
            dirname, basename, ext = path_decompose(img_path)
            img_paths.append(img_path)
            data_paths.append(os.path.join(dataset_dir, "data", f"{basename}.npy"))

        self.img_paths = img_paths
        self.data_paths = data_paths

        self.transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.label_resize = size

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        data_path = self.data_paths[index]

        img = Image.open(img_path)

        iw = img.width
        ih = img.height
        img = self.transform(img)

        data = np.load(data_path, allow_pickle=True).item()
        s = 8
        label = np.zeros((s, s, 5), dtype=np.float32)
        for x, y, w, h in data["boxes"]:
            x, y, w, h = x / iw, y / ih, w / iw, h / ih
            cx = x + w * 0.5
            cy = y + h * 0.5
            label[int(cy * s), int(cx * s), 0] = 1
            label[int(cy * s), int(cx * s), 1:3] = [
                cx * s - int(cx * s),
                cy * s - int(cy * s),
            ]
            label[int(cy * s), int(cx * s), 3:] = [w, h]

        # 全变成0
        # label = np.ones((s, s, 5), dtype=np.float32)
        # label[:,s//2:,:]=0

        label = torch.from_numpy(label)

        return img, label, img_path

    def __len__(self):
        return len(self.img_paths)


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
        # "train_dataset_dir": "/data_ssd/CocoDetection",
        "train_dataset_dir": "/data_ssd/supervislyDetection",
        "val_dataset_dir": "/data_ssd/valDetection",
        "checkpoint_dir": "/checkpoint/detection_20200622",
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

    size = (256, 256)

    trainset = DetectionDataset(args.train_dataset_dir, size)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.cpu_num
    )

    valset = DetectionDataset(args.val_dataset_dir, size)

    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    model = Detect()
    # optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    start_epoch = 0
    if args.continue_train:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.epoch_model}.pth")
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

    criterion = Loss()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if show_img_tag:
        window_name = "img | mix"
        show_img = None

    print("training...")
    for epoch in range(args.epoch):

        if args.continue_train and epoch < start_epoch:
            continue

        loss_total = 0.0
        for i0, (inputs, labels, img_paths) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
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

                train_acc_fore, train_acc_back, train_x, train_y, train_w, train_h = box_eval(outputs, labels)
                with torch.no_grad():
                    val_cf = 0
                    val_cb = 0
                    val_x = 0
                    val_y = 0
                    val_w = 0
                    val_h = 0

                    for j0, (inputs2, labels2, _) in enumerate(valloader):
                        inputs2, labels2 = inputs2.to(device), labels2.to(device)
                        outputs = model(inputs2)

                        val_acc_fore,val_acc_back, ax, ay, aw, ah = box_eval(outputs, labels2)
                        val_cf += val_acc_fore
                        val_cb += val_acc_back
                        val_x += ax
                        val_y += ay
                        val_w += aw
                        val_h += ah

                    val_cf = val_cf / len(valloader)
                    val_cb = val_cb / len(valloader)
                    val_x = val_x / len(valloader)
                    val_y = val_y / len(valloader)
                    val_w = val_w / len(valloader)
                    val_h = val_h / len(valloader)

                    print(
                        f" [epoch {epoch}]"
                        + f" [val_num:{len(valset)}]\n"
                        + f" [train_acc_fore: {round(train_acc_fore, 6)}]"
                        + f" [train_acc_back: {round(train_acc_back, 6)}]"
                        + f" [train_x: {round(train_x,6)}]"
                        + f" [train_y: {round(train_y,6)}]"
                        + f" [train_w: {round(train_w,6)}]"
                        + f" [train_h: {round(train_h,6)}]\n"
                        + f" [  val_acc_fore: {round(val_cf, 6)}]"
                        + f" [  val_acc_back: {round(val_cb, 6)}]"
                        + f" [  val_x: {round(val_x,6)}]"
                        + f" [  val_y: {round(val_y,6)}]"
                        + f" [  val_w: {round(val_w,6)}]"
                        + f" [  val_h: {round(val_h,6)}]"
                    )

                    img_tensor, data_val, img_path = random.choice(valset)

                    img = cv.imread(img_path)
                    img = cv.resize(img, size)

                    mix = img.copy()

                    output = model(img_tensor[np.newaxis, :].to(device))
                    output[:,:,:,0] = torch.sigmoid(output[:,:,:,0] )
                    output = output.cpu().numpy()[0]

                    for iy, ix in product(
                        range(output.shape[0]), range(output.shape[1])
                    ):
                        if output[iy, ix, 0] < 0.5:
                            continue

                        ny, nx = output.shape[:2]

                        cx, cy, cw, ch = output[iy, ix, 1:]
                        x = (ix + cx) * size[1] / nx
                        y = (iy + cy) * size[1] / ny
                        w = cw * size[0] / nx
                        h = ch * size[0] / ny

                        cv.rectangle(
                            mix,
                            (int(x), int(y)),
                            (int(x + w + 1), int(y + h + 1)),
                            (0, 255, 0),
                            1,
                        )

                    show_img = np.concatenate([img, mix], axis=1)

                model.train()

            if show_img_tag and show_img is not None:
                cv.imshow(window_name, show_img)
                cv.waitKey(5)

        continue

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
