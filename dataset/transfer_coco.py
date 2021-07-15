import cv2 as cv
import numpy as np
import json
from ymtools.common_dataset_api import key_combine
from ymtools.common import path_decompose
from shutil import copyfile
import ochumanApi.vis as vistool
import tqdm


def get_body_keypoint(kpt):
    kpt = np.array(kpt, dtype=np.int32).reshape(-1, 3)
    kpt = np.array(kpt)
    npart = kpt.shape[0]

    if npart == 17:  # coco
        part_names = ['nose',
                      'left_eye', 'right_eye', 'left_ear', 'right_ear',
                      'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                      'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                      'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        # visible_map = {1: 'vis',
        #               2: 'not_vis',
        #               3: 'missing'}
        visible_map = {2: 'vis',
                       1: 'not_vis',
                       0: 'missing'}
        map_visible = {value: key for key, value in visible_map.items()}
        # if connection is None:
        connection = [[16, 14], [14, 12], [17, 15],
                      [15, 13], [12, 13], [6, 12],
                      [7, 13], [6, 7], [6, 8],
                      [7, 9], [8, 10], [9, 11],
                      [2, 3], [1, 2], [1, 3],
                      [2, 4], [3, 5], [4, 6], [5, 7]]
    elif npart == 19:  # ochuman
        part_names = ["right_shoulder", "right_elbow", "right_wrist",
                      "left_shoulder", "left_elbow", "left_wrist",
                      "right_hip", "right_knee", "right_ankle",
                      "left_hip", "left_knee", "left_ankle",
                      "head", "neck"] + \
            ['right_ear', 'left_ear', 'nose', 'right_eye', 'left_eye']

        visible_map = {0: 'missing',
                       1: 'vis',
                       2: 'self_occluded',
                       3: 'others_occluded'}

        map_visible = {value: key for key, value in visible_map.items()}
        connection = [[16, 19], [13, 17], [4, 5],
                      [19, 17], [17, 14], [5, 6],
                      [17, 18], [14, 4], [1, 2],
                      [18, 15], [14, 1], [2, 3],
                      [4, 10], [1, 7], [10, 7],
                      [10, 11], [7, 8], [11, 12], [8, 9],
                      [16, 4], [15, 1]]  # TODO

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
              [255, 255, 0], [170, 255, 0], [85, 255, 0],
              [0, 255, 0], [0, 255, 85], [0, 255, 170],
              [0, 255, 255], [0, 170, 255], [0, 85, 255],
              [0, 0, 255], [85, 0, 255], [170, 0, 255],
              [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    body_keypoint = {}

    for i0, vs in enumerate(kpt):
        x, y, v = vs

        key = part_names[i0]

        one_key = {}
        if len(part_names) == 19:
            key_map = {
                0: 'missing',
                1: 'vis',
                2: 'not_vis',
                3: 'not_vis'
            }
        else:
            key_map = {
                0: 'missing',
                1: 'not_vis',
                2: 'vis',
            }

        one_key[key_combine('status', 'keypoint_status')] = key_map[v]
        one_key[key_combine('point', 'point_xy')] = [int(x), int(y)]

        body_keypoint[key_combine(key, 'sub_dict')] = one_key

    return body_keypoint


def transfer_coco(img_dir, ann_path, save_dir):

    colors = [
        [255, 0, 0],
        [255, 255, 0],
        [0, 255, 0],
        [0, 255, 255],
        [0, 0, 255],
        [255, 0, 255]]

    from pycocotools.coco import COCO

    coco = COCO(ann_path)

    catIds = coco.getCatIds(catNms=['person'])

    imgIds = coco.getImgIds(catIds=catIds)

    imgs = coco.loadImgs(imgIds)

    for j0, imgd in enumerate(tqdm.tqdm(imgs)):

        nd = {}

        meta = {}

        filename = imgd['file_name']
        _, name, ext = path_decompose(imgd['file_name'])

        save_image_dir = os.path.join(save_dir, 'image')
        instance_dir = os.path.join(save_dir, 'instance_mask')
        instance_name_dir = os.path.join(save_dir, 'instance_mask', name)
        segment_mask_dir = os.path.join(save_dir, 'segment_mask')
        class_mask_dir = os.path.join(save_dir, 'class_mask')
        class_mask_name_dir = os.path.join(save_dir, 'class_mask', name)
        mix_dir = os.path.join(save_dir, 'mix')
        data_dir = os.path.join(save_dir, 'data')

        for dir in [save_dir, save_image_dir, instance_dir, segment_mask_dir,
                    class_mask_dir, class_mask_name_dir, mix_dir, data_dir, instance_name_dir]:
            if not os.path.exists(dir):
                os.mkdir(dir)

        load_image_path = os.path.join(img_dir, filename)
        save_image_path = os.path.join(save_image_dir, filename)

        copyfile(load_image_path, save_image_path)
        nd[key_combine('image', 'image_path')] = os.path.join(
            'image', filename)

        img = cv.imread(load_image_path)
        meta['origin_image_path'] = load_image_path
        h, w = imgd['height'], imgd['width']
        meta['width'] = w
        meta['height'] = h

        class_name = 'person'

        nd[key_combine('meta', 'other')] = meta

        nd[key_combine('class', 'class')] = class_name

        annId = coco.getAnnIds(imgIds=imgd['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annId)

        img_show = img.copy()
        segment_mask = np.zeros((h, w), dtype=np.uint8)

        objs = []

        for i0, anno in enumerate(anns):
            obj = {}

            bbox = anno['bbox']
            kpt = anno['keypoints']
            segm = anno['segmentation']

            bbox = [int(bbox[0]), int(bbox[1]), int(
                bbox[0]+1+bbox[2]), int(bbox[1]+1+bbox[3])]

            obj[key_combine('box', 'box_xyxy')] = bbox
            obj[key_combine('class', 'class')] = class_name

            img_show = vistool.draw_bbox(
                img_show, bbox, thickness=2, color=colors[i0 % len(colors)])

            if segm is not None:
                instance_mask = (coco.annToMask(anno)*255).astype(np.uint8)

                segment_mask |= instance_mask

                cv.imwrite(os.path.join(
                    instance_name_dir, str(i0)+'.png'), instance_mask)
                obj[key_combine('instance_mask', 'mask_path')] = os.path.join(
                    'instance_mask', name, str(i0)+'.png')

                img_show = vistool.draw_mask(
                    img_show, instance_mask, thickness=3, color=colors[i0 % len(colors)])

            if kpt is not None:
                img_show = vistool.draw_skeleton(
                    img_show, kpt, connection=None, colors=colors[i0 % len(colors)], bbox=bbox)

                body_keypoint = get_body_keypoint(kpt)

                obj[key_combine('body_keypoint', 'sub_dict')] = body_keypoint

            objs.append(obj)

        cv.imwrite(os.path.join(mix_dir, filename), img_show)
        nd[key_combine('mix', 'image_path')] = os.path.join('mix', filename)

        cv.imwrite(os.path.join(segment_mask_dir, name+'.png'), segment_mask)
        nd[key_combine('segment_mask', 'mask_path')] = os.path.join(
            'segment_mask', name+'.png')

        copyfile(os.path.join(segment_mask_dir, name+'.png'),
                 os.path.join(class_mask_dir, name, 'person.png'))

        class_masks = []
        class_mask = {}
        class_mask[key_combine('class', 'class')] = 'person'
        class_mask[key_combine('segment_mask', 'mask_path')] = os.path.join(
            'class_mask', name, 'person.png')
        class_masks.append(class_mask)
        nd[key_combine('class_mask', 'sub_list')] = class_masks

        nd[key_combine('object', 'sub_list')] = objs

        nd_json = json.dumps(nd)
        with open(os.path.join(data_dir, name+'.json'), 'w') as f:
            f.write(nd_json)

        pass

    print(catIds)


if __name__ == '__main__':
    from ymtools.debug_function import *
    transfer_coco(
        '/Users/yanmiao/yanmiao/data/coco/val2017',
        '/Users/yanmiao/yanmiao/data/coco/annotations/person_keypoints_val2017.json',
        '/Users/yanmiao/yanmiao/data-common/coco'
    )
