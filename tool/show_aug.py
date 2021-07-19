from ymlib.common_dataset_api import common_ann_loader, common_aug, common_choice, common_filter, common_transfer, key_combine
from ymlib.dataset_visual import mask2box, draw_box, draw_keypoint, draw_mask, draw_label
import imgaug as ia
from imgaug import augmenters as iaa
import cv2 as cv
import time
from ymlib.debug_function import *

ORDER_PART_NAMES = ["right_shoulder", "right_elbow", "right_wrist",
                    "left_shoulder", "left_elbow", "left_wrist",
                    "right_hip", "right_knee", "right_ankle",
                    "left_hip", "left_knee", "left_ankle",
                    "head", "neck", 'right_ear', 'left_ear',
                    'nose', 'right_eye', 'left_eye']


def test1(dataset_dir):

    for ann in common_ann_loader(dataset_dir):

        common_choice(ann, key_choices={'image', 'object'})

        common_transfer(ann)
        image = ann[key_combine('image', 'image')]

        objs = ann[key_combine('object', 'sub_list')]

        for obj in objs:
            start = time.time()

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

            box = obj[key_combine('box', 'box_xyxy')]
            class_name = obj[key_combine('class', 'class')]

            common_choice(obj, key_choices={'instance_mask', 'body_keypoint'})

            common_transfer(obj)

            obj[key_combine('image', 'image')] = image

            # aug过程

            def sometimes(x): return iaa.Sometimes(1.0, x)

            ih, iw = image.shape[:2]
            x0, y0, x1, y1 = box
            box_center_x = (x0+x1)/2
            box_center_y = (y0+y1)/2
            tx = int(iw / 2 - box_center_x)
            ty = int(ih / 2 - box_center_y)

            aug = iaa.Sequential([
                iaa.Affine(translate_px={"x": (tx, tx), "y": (ty, ty)}),
                # sometimes(
                #     iaa.Affine(rotate=(-25, 25)),
                # ),
            ])

            common_aug(obj, aug, r=True)

            instance_mask = obj[key_combine('instance_mask', 'mask')]
            instance_box = mask2box(instance_mask)
            if instance_box is None:
                continue
            x1, y1, x2, y2 = instance_box
            pad = 16
            left = -x1 + pad
            right = x2 - iw + pad
            top = -y1 + pad
            bottom = y2 - ih + pad
            # aw = int((x2-x1)*0.2)
            # ah = int((y2-y1)*0.2)
            aw = 0
            ah = 0

            aug2 = iaa.Sequential([
                iaa.CropAndPad(((top-ah, top+ah), (right-aw, right+aw),
                               (bottom-ah, bottom+ah), (left-aw, left+aw))),
                # iaa.Fliplr(0.5),
                # sometimes(iaa.LinearContrast((0.75, 1.5))),
                # sometimes(iaa.AdditiveGaussianNoise(
                #     loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                # sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.2)),
            ])

            common_aug(obj, aug2, r=True)

            # 数据集显示

            instance_mask = obj[key_combine('instance_mask', 'mask')]
            instance_image = obj[key_combine('image', 'image')]
            instance_mix = instance_image.copy()

            draw_mask(instance_mix, instance_mask)

            instance_box = mask2box(instance_mask)
            draw_box(instance_mix, instance_box)

            if key_combine('body_keypoint', 'sub_dict') in obj:
                body_keypoint = obj[key_combine('body_keypoint', 'sub_dict')]
                draw_keypoint(instance_mix, body_keypoint, labeled=True)

            draw_label(instance_mix, class_name, instance_box[:2], thickness=2)

            h, w = image.shape[:2]
            window_name = f'image | mix | mask   height:{h} width:{w}   time:{int((time.time() - start)*1000)}'
            instance_mask = cv.cvtColor(instance_mask, cv.COLOR_GRAY2RGB)

            imshow(np.concatenate(
                [instance_image, instance_mix, instance_mask],
                axis=1), window_name)


def show_dataset(dataset_dir):
    aug = iaa.Noop()

    for ann in common_ann_loader(dataset_dir):
        common_choice(ann, key_choices={
                      'image', 'mix', 'segment_mask', 'meta', 'object'})

        def filter(result):
            yield True

        if not common_filter(ann, filter):
            continue

        start = time.time()

        common_transfer(ann)
        common_aug(ann, aug)

        image = ann[key_combine('image', 'image')]
        origin_image_path = ann[key_combine(
            'meta', 'other')]['origin_image_path']

        h, w = image.shape[:2]
        window_name = f'image | mix | mask   height:{h} width:{w} origin:{origin_image_path}  time:{int((time.time() - start)*1000)}'

        # mix = ann[key_combine('mix', 'image')]

        mask = ann[key_combine('segment_mask', 'mask')]

        mix = image.copy()

        mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)

        objs = ann[key_combine('object', 'sub_list')]

        for obj in objs:
            common_transfer(obj)
            if key_combine('body_keypoint', 'sub_dict') in obj:
                body_keypoint = obj[key_combine('body_keypoint', 'sub_dict')]
                draw_keypoint(mix, body_keypoint, labeled=True)

            instance_mask = obj[key_combine('instance_mask', 'mask')]

            draw_mask(mix,instance_mask)

        imshow(np.concatenate([image, mix, mask], axis=1), window_name)


if __name__ == '__main__':

    dataset_dir = '/Users/yanmiao/yanmiao/data-common/hun_sha_di_pian'
    # dataset_dir = '/Users/yanmiao/yanmiao/data-common/ochuman'
    # dataset_dir = '/Users/yanmiao/yanmiao/data-common/supervisely'
    # dataset_dir = '/Users/yanmiao/yanmiao/data-common/coco'

    show_dataset(dataset_dir)
    # test1(dataset_dir)
    # test2()
