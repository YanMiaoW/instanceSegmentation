import json
import cv2 as cv
import os
import tqdm
import numpy as np
from ymtools.common_dataset_api import key_combine, BODY_PART_CHOICES, CLASS
from shutil import copyfile


def transfer_supervisely_to_common(data_dir, save_dir):
    import supervisely_lib as sly

    project = sly.Project(data_dir, sly.OpenMode.READ)

    pbar = tqdm.tqdm(total=project.total_items)
    i0 = 0
    for dataset in project:
        for item_name in dataset:
            item_paths = dataset.get_item_paths(item_name)

            pbar.update(1)
            # if pbar.n not in  [3000,3400,3500,3600]:
            #     continue
            with open(item_paths.ann_path, 'r') as f:
                sann_json = json.load(f)

            sann = sly.Annotation.load_json_file(
                item_paths.ann_path, project.meta)

            for label in sann.labels:
                if label.obj_class.name not in [
                    'person_poly', 'person_bmp', 'neutral', 'persona',
                        *list(BODY_PART_CHOICES)]:
                    print(label.obj_class.name)
                    assert False, 'not support some obj class name'

            def class2common(class_str):

                if class_str in ['person_poly', 'person_bmp', 'persona']:
                    return 'person'

                elif any(class_str in s for s in [CLASS, BODY_PART_CHOICES]):
                    return class_str

                else:
                    return None

            ann = {}

            meta = {}

            cls_masks = []
            cls_mask = {}

            meta['origin_image_path'] = item_paths.img_path

            name = str(i0).zfill(5)

            seg_class_name = 'person'

            # if os.path.exists(os.path.join(save_dir, 'instance_mask', name)):
            #     i0 += 1
            #     continue

            image_dir = os.path.join(save_dir, 'image')
            instance_dir = os.path.join(save_dir, 'instance_mask')
            instance_name_dir = os.path.join(save_dir, 'instance_mask', name)
            segment_mask_dir = os.path.join(save_dir, 'segment_mask')
            class_mask_dir = os.path.join(save_dir, 'class_mask')
            class_mask_name_dir = os.path.join(save_dir, 'class_mask', name)
            mix_dir = os.path.join(save_dir, 'mix')
            data_dir = os.path.join(save_dir, 'data')

            for dir in [save_dir, image_dir, instance_dir, segment_mask_dir,
                        class_mask_dir, class_mask_name_dir, mix_dir, data_dir, instance_name_dir]:
                if not os.path.exists(dir):
                    os.mkdir(dir)

            img = cv.imread(item_paths.img_path)
            h, w = img.shape[:2]

            meta['width'] = w
            meta['height'] = h

            image_path = os.path.join("image", name+'.png')
            cv.imwrite(os.path.join(save_dir, image_path), img)

            ann[key_combine('image', 'image_path')] = image_path
            ann[key_combine('class', 'class')] = 'person'

            mask = np.zeros((h, w), dtype=np.uint8)

            mix = img.copy()
            sann.draw(mix)

            j0 = 0
            objs = {}

            for i, (label, obj_json) in enumerate(zip(sann.labels, sann_json['objects'])):
                instance_id = obj_json['instance']
                c = class2common(label.obj_class.name)
                if c is None:
                    continue

                if instance_id not in objs:
                    objs[instance_id] = {}
                    objs[instance_id][key_combine(
                        'body_keypoint', 'sub_dict')] = {}

                if c in BODY_PART_CHOICES:
                    xy = obj_json['points']['exterior'][0]
                    keypoint = {}
                    keypoint[key_combine('status', 'keypoint_status')] = 'vis'
                    keypoint[key_combine('point', 'point_xy')] = xy

                    objs[instance_id][key_combine(
                        'body_keypoint', 'sub_dict')][key_combine(c, 'sub_dict')] = keypoint

                if c in CLASS:
                    obj = objs[instance_id]
                    instance = np.zeros((h, w, 3), dtype=np.uint8)
                    label.draw(instance, color=[255, 255, 255])
                    instance = instance[:, :, 0]
                    if c == 'person':
                        mask |= instance

                    b = label.geometry.to_bbox()
                    x0, y0, x1, y1 = b.left, b.top, b.right, b.bottom

                    instance_path = os.path.join(
                        'instance_mask', name, str(j0)+'.png')
                    j0 += 1
                    cv.imwrite(os.path.join(save_dir, instance_path), instance)

                    obj[key_combine('instance_mask', 'mask_path')
                        ] = instance_path
                    obj[key_combine('box', 'box_xyxy')] = [x0, y0, x1, y1]
                    obj[key_combine('class', 'class')] = c
                    # objs[]

                cv.rectangle(mix, (int(x0), int(y0)),
                             (int(x1 + 1), int(y1 + 1)), (0, 255, 0), 1)

            segment_mask_path = os.path.join('segment_mask', name+'.png')
            cv.imwrite(os.path.join(save_dir, segment_mask_path), mask)

            mix_path = os.path.join('mix', name+'.png')
            cv.imwrite(os.path.join(save_dir, mix_path), mix)
            ann[key_combine('mix', 'image_path')] = mix_path

            ann[key_combine('segment_mask', 'mask_path')] = segment_mask_path

            class_mask_path = os.path.join(
                'class_mask', name, seg_class_name+'.png')
            copyfile(os.path.join(save_dir, segment_mask_path),
                     os.path.join(save_dir, class_mask_path))

            cls_mask[key_combine('class', 'class')] = seg_class_name

            cls_mask[key_combine('segment_mask', 'mask_path')
                     ] = class_mask_path

            cls_masks.append(cls_mask)
            ann[key_combine('class_mask', 'sub_list')] = cls_masks
            ann[key_combine('meta', 'other')] = meta
            ann[key_combine('object', 'sub_list')] = list(objs.values())

            data_path = os.path.join('data', name+'.json')
            ann_json = json.dumps(ann)
            with open(os.path.join(save_dir, data_path), 'w') as f:
                f.write(ann_json)

            i0 += 1

    print()
    pbar.close()


if __name__ == "__main__":
    # 示例
    from ymtools.debug_function import *
    transfer_supervisely_to_common(
        # '/data/SuperviselyPeopleDatasets',
        # '/Users/yanmiao/yanmiao/data/hun_sha_di_pian/labeled/hun',
        '/Users/yanmiao/yanmiao/data/hun_sha_di_pian/addkeypoint/hun',
        '/Users/yanmiao/yanmiao/data-common/hun_sha_di_pian'
        # '/data_ssd/supervisely'
        # '/data_ssd/val'
    )
