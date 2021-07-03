import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import tqdm
import json
import cv2 as cv
from debug_function import *


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="generate train people segmentaion")
    parser.add_argument(
        "-o", "--output-dir", help="train dataset save dir", required=True
    )
    parser.add_argument(
        "--continue-generate", action="store_true", help="skip generate file."
    )
    args = parser.parse_args()
    return args


def path_decompose(path):
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)
    ext = os.path.splitext(path)[-1][1:]
    basename = os.path.splitext(basename)[0]
    return dirname, basename, ext


def cocoHuman():
    from pycocotools.coco import COCO
    import pycocotools.mask as maskUtils

    json_path = "/data/coco2017/annotations/instances_train2017.json"
    # json_path = "/data/coco2017/annotations/instances_val2017.json"
    # json_path = "/data/coco2017/annotations/instances_small_val2017_human.json"
    image_dir = "/data/coco2017/train2017"
    # image_dir = "/data/coco2017/val2017"

    coco = COCO(json_path)
    catIds = coco.getCatIds(catNms=["person"])  # 获取指定类别 id

    imgIds = coco.getImgIds(catIds=catIds)  # 获取图片i
    imgDatas = coco.loadImgs(
        imgIds
    )  # 加载图片,loadImgs() 返回的是只有一个内嵌字典元素的list, 使用[0]来访问这个元素
    # image = io.imread(train_path + img['file_name'])

    for img_data in tqdm.tqdm(imgDatas):
        annIds = coco.getAnnIds(imgIds=[img_data["id"]], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        image = cv.imread(os.path.join(image_dir, img_data["file_name"]))
        ih, iw = image.shape[:2]

        image_show = image.copy()

        boxes = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, w, h])
            cv.rectangle(
                image_show,
                (int(x), int(y)),
                (int(x + w + 1), int(y + h + 1)),
                (0, 255, 0),
                2,
            )

        data = {"boxes": boxes}
        filename = f'coco_{img_data["id"].zfill(6)}'
        np.save(os.path.join(args.data_dir, filename + ".npy"), data)
        cv.imwrite(os.path.join(args.image_dir, filename + ".jpg"), image)
        cv.imwrite(os.path.join(args.mix_dir, filename + ".jpg"), image_show)


def supervisly():
    import supervisely_lib as sly

    project = sly.Project("/data/SuperviselyPeopleDatasets", sly.OpenMode.READ)

    classes_filter = ["person_poly", "person_bmp"]

    pbar = tqdm.tqdm(total=project.total_items)
    for dataset in project:
        for item_name in dataset:
            # 更新进度条
            pbar.update(1)

            item_paths = dataset.get_item_paths(item_name)

            ann = sly.Annotation.load_json_file(item_paths.ann_path, project.meta)
            ann.get_label_by_id()

            for label in ann.labels:
                if label.obj_class.name in classes_filter:
                    b = label.geometry.to_bbox()
                    x0, y0, x1, y1 = b.left, b.top, b.right, b.bottom
                    x, y, w, h = x0, y0, x1 - x0, y1 - y0
                    

                    img_path = os.path.join(dataset.img_dir, item_name)
                    image = cv.imread(img_path)
                    
                    c = 256 / min(image.shape[:2])
                    
                    image = cv.resize(image,None,fx=c,fy=c)
                    x,y,w,h = x*c ,y*c,w*c,h*c
                    
                    image_show = image.copy()

                    cv.rectangle(
                        image_show,
                        (int(x), int(y)),
                        (int(x + w + 1), int(y + h + 1)),
                        (0, 255, 0),
                        1,
                    )
                    
                    data={
                        'boxes':[[x,y,w,h]]
                    }

                    dirname, name, ext = path_decompose(img_path)
                    
                    

                    filename = f"sup_{dataset.name}_{name}"
                    np.save(
                        os.path.join(args.data_dir, filename + ".npy"),data
                    )
                    cv.imwrite(os.path.join(args.image_dir, filename + ".png"), image)
                    cv.imwrite(
                        os.path.join(args.mix_dir, filename + ".png"),
                        image_show,
                    )

    pbar.close()


if __name__ == "__main__":
    args = parse_args()
    args.data_dir = os.path.join(args.output_dir, "data")
    args.image_dir = os.path.join(args.output_dir, "image")
    args.mix_dir = os.path.join(args.output_dir, "mix")

    for path in [args.output_dir, args.data_dir, args.image_dir, args.mix_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    # cocoHuman()
    supervisly()
