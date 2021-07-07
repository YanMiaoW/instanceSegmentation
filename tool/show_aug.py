from dataset.common_dataset_api import common_ann_loader, common_aug, common_choice, common_filter, common_transfer, key_combine
from dataset.dataset_visual import mask2box, draw_box, draw_keypoint, draw_mask, draw_label
import imgaug as ia
from imgaug import augmenters as iaa
import cv2 as cv
import time
from debug_function import *


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

            common_choice(obj, key_choices={
                'instance_image', 'instance_mask', 'body_keypoint'})

            common_transfer(obj)

            obj[key_combine('instance_image', 'image')] = image

            # aug过程

            def sometimes(x): return iaa.Sometimes(1.0, x)

            ih, iw = image.shape[:2]
            x0, y0, x1, y1 = box
            box_center_x = (x0+x1)/2
            box_center_y = (y0+y1)/2
            tx = int(iw / 2 - box_center_x)
            ty = int(ih / 2 - box_center_y)

            common_aug(obj, iaa.Sequential([
                iaa.Affine(translate_px={"x": (tx, tx), "y": (ty, ty)}),
                sometimes(
                    iaa.Affine(rotate=(-25, 25)),
                ),
            ]), r=True)

            instance_mask = obj[key_combine('instance_mask', 'mask')]
            instance_box = mask2box(instance_mask)
            if instance_box is None:
                continue
            x1, y1, x2, y2 = instance_box
            left = -x1
            right = x2 - iw
            top = -y1
            bottom = y2 - ih
            aw = int((x2-x1)*0.2)
            ah = int((y2-y1)*0.2)

            aug = iaa.Sequential([
                iaa.CropAndPad(((top-ah, top+ah), (right-aw, right+aw),
                               (bottom-ah, bottom+ah), (left-aw, left+aw))),
                iaa.Fliplr(0.5),
                sometimes(iaa.LinearContrast((0.75, 1.5))),
                sometimes(iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.2)),
            ])

            common_aug(obj, aug, r=True)

            # 数据集显示

            instance_mask = obj[key_combine('instance_mask', 'mask')]
            instance_image = obj[key_combine('instance_image', 'image')]
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


def test2():
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmentables import Keypoint, KeypointsOnImage

    ia.seed(1)

    image = ia.quokka(size=(256, 256))
    kps = KeypointsOnImage([
        Keypoint(x=65, y=100),
        Keypoint(x=75, y=200),
        Keypoint(x=100, y=100),
        Keypoint(x=200, y=80)
    ], shape=image.shape)

    seq1 = iaa.Affine(translate_px={"x": (200, 200), "y": (0, 0)})

    seq = iaa.Sequential([
        # change brightness, doesn't affect keypoints
        iaa.Multiply((1.2, 1.5)),
        iaa.Fliplr(),

        iaa.Affine(
            translate_px={"x": (100, 100), "y": (0, 0)},
            rotate=10,
            scale=(0.5, 0.7)
        )  # rotate by exactly 10deg and scale to 50-70%, affects keypoints
    ])

    # Augment keypoints and images.
    image_aug, kps_aug = seq1(image=image, keypoints=kps)

    # image_aug, kps_aug = seq(image=image_aug, keypoints=kps_aug)
    image_aug, kps_aug = seq(image=image, keypoints=kps)

    # print coordinates before/after augmentation (see below)
    # use after.x_int and after.y_int to get rounded integer coordinates
    for i in range(len(kps.keypoints)):
        before = kps.keypoints[i]
        after = kps_aug.keypoints[i]
        print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
            i, before.x, before.y, after.x, after.y)
        )

    # image with keypoints before/after augmentation (shown below)
    image_before = kps.draw_on_image(image, size=7)
    image_after = kps_aug.draw_on_image(image_aug, size=7)
    imshow(np.concatenate([image_before, image_after], axis=1))


if __name__ == '__main__':

    # dataset_dir = '/Users/yanmiao/yanmiao/data-common/hun_sha_di_pian'
    dataset_dir = '/Users/yanmiao/yanmiao/data-common/ochuman'
    # dataset_dir = '/Users/yanmiao/yanmiao/data-common/supervisely'
    # dataset_dir = '/Users/yanmiao/yanmiao/data-common/coco'

    # show_dataset(dataset_dir)
    test1(dataset_dir)
    # test2()
