from dataset.common_dataset_api import common_ann_loader, common_aug, common_choice, common_filter, common_transfer, key_combine
import imgaug as ia
from imgaug import augmenters as iaa
import cv2 as cv


def test_aug():
    return iaa.Sequential([
        # iaa.PiecewiseAffine(scale=(0.01, 0.05)),  #仿射
        #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05),per_channel=0.2),
        #iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
        # iaa.Superpixels(p_replace=(0, 1.0),n_segments=(20, 200)),  #超像素
        # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  #锐化
        iaa.Crop(px=(0, 300), keep_size=False),  # 裁剪
        iaa.Fliplr(0.5),  # 翻转
        iaa.GaussianBlur(sigma=(0, 3.0))  # 高斯模糊
    ])


if __name__ == '__main__':
    from debug_function import *
    import time

    # aug = test_aug()
    aug = iaa.Noop()
    # dataset_dir = '/Users/yanmiao/yanmiao/data-common/supervisely'
    dataset_dir = '/Users/yanmiao/yanmiao/data-common/coco'
    for ann in common_ann_loader(dataset_dir):
        common_choice(ann, key_choices={'image', 'mix', 'segment_mask'})

        def filter(result):
            yield True

        if not common_filter(ann, filter):
            continue

        start = time.time()

        common_transfer(ann)
        common_aug(ann, aug)

        image = ann[key_combine('image', 'image')]
        h, w = image.shape[:2]
        window_name = f'image | mix | mask   height:{h} width:{w}   time:{int((time.time() - start)*1000)}'
        mix = ann[key_combine('mix', 'image')]
        mask = ann[key_combine('segment_mask', 'mask')]
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)

        imshow(np.concatenate([image, mix, mask], axis=1), window_name)
