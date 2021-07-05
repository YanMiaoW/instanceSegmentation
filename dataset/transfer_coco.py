import cv2 as cv
import numpy as np

def transfer_coco(ann_path, save_dir):
    from pycocotools.coco import COCO

    coco=COCO(ann_path)

    catIds = coco.getCatIds()

    print()


if __name__ == '__main__':
    from debug_function import *
    transfer_coco(
        '/Users/yanmiao/yanmiao/data/ochuman/annotations/ochuman.json',
        '/Users/yanmiao/yanmiao/data-common/coco'
    )
