import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import tqdm
import json
import cv2 as cv
import argparse
from debug_function import *


def parse_args():
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


def ochuman():

    import pycocotools.mask as maskUtils

    # 由轮廓绘制mask
    def _poly2mask(mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann["counts"], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    json_path = "/data/OCHuman/annotations/ochuman.json"
    image_path = "/data/OCHuman/images"

    with open(json_path) as f:
        json_data = json.load(f)["images"]

    for image_data in tqdm.tqdm(json_data):
        # 逐一遍历各图
        image_name = image_data["file_name"]
        image = cv.imread(os.path.join(image_path, image_name))
        image_show = image.copy()
        h, w = image.shape[:2]
        name = image_name.split(".")[0]

        # 遍历单图中所有实例
        for i, person in enumerate(image_data["annotations"]):
            seg = person["segms"]  # 分割信息

            if seg is not None:
                mask = _poly2mask(seg["outer"], h, w)  # 获得单个实例分割mask
                if mask.shape[:2] != image.shape[:2]:
                    continue
                if seg["inner"] is not None and len(seg["inner"]):  # 内圈抠除
                    mask_zeros = _poly2mask(seg["inner"], h, w)
                    mask = np.bitwise_and(mask, 255 - mask_zeros)
                mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
                # image_show仅用于实例展示
                image_show[mask > 0] = (
                    np.array([0, 255, 255], dtype=np.uint8) // 2
                    + image_show[mask > 0] // 2
                )

                locs = np.where(mask > 0)
                x0 = np.min(locs[1])
                x1 = np.max(locs[1])
                y0 = np.min(locs[0])
                y1 = np.max(locs[0])

                mask = mask[y0 : y1 + 1, x0 : x1 + 1]
                image = image[y0 : y1 + 1, x0 : x1 + 1]
                image_show = image_show[y0 : y1 + 1, x0 : x1 + 1]

                # 保存重组后的信息
                cv.imwrite(os.path.join(args.mask_dir, "qc_%s.png" % name), mask)
                cv.imwrite(os.path.join(args.image_dir, "qc_%s.png" % name), image)
                cv.imwrite(os.path.join(args.mix_dir, "qc_%s.png" % name), image_show)


def supervisely():
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

            for label in ann.labels:
                if label.obj_class.name in classes_filter:
                    b = label.geometry.to_bbox()
                    x0, y0, x1, y1 = b.left, b.top, b.right, b.bottom

                    mask = np.zeros(ann.img_size, dtype=np.uint8)
                    label.draw(mask, color=255)

                    img_path = os.path.join(dataset.img_dir, item_name)
                    image = cv.imread(img_path)

                    image_show = image.copy()
                    image_show[mask > 0] = (
                        np.array([0, 255, 255], dtype=np.uint8) // 2
                        + image_show[mask > 0] // 2
                    )

                    mask = mask[y0 : y1 + 1, x0 : x1 + 1]
                    image = image[y0 : y1 + 1, x0 : x1 + 1]
                    image_show = image_show[y0 : y1 + 1, x0 : x1 + 1]

                    dirname, name, ext = path_decompose(img_path)

                    # 保存重组后的信息
                    cv.imwrite(os.path.join(args.mask_dir, "sup_%s.png" % name), mask)
                    cv.imwrite(os.path.join(args.image_dir, "sup_%s.png" % name), image)
                    cv.imwrite(
                        os.path.join(args.mix_dir, f"sup_{dataset.name}_{name}.png"), image_show
                    )

    pbar.close()


if __name__ == "__main__":
    args = parse_args()
    args.mask_dir = os.path.join(args.output_dir, "mask")
    args.image_dir = os.path.join(args.output_dir, "image")
    args.mix_dir = os.path.join(args.output_dir, "mix")

    for path in [args.output_dir, args.mask_dir, args.image_dir, args.mix_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    # ochuman()
    supervisely()
