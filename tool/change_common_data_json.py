from dataset.common_dataset_api import key_combine
import os
import json
import glob
import tqdm


if __name__ == '__main__':
    dataset_dir = '/Users/yanmiao/yanmiao/data-common/supervisely'

    ann_paths = glob.glob(os.path.join(dataset_dir, 'data', '*.json'))
    for ann_path in tqdm.tqdm(ann_paths):
        with open(ann_path) as f:
            ann = json.load(f)

        # ann[key_combine('class','class')] = ann[key_combine('class','other')]
        # del ann[key_combine('class','other')]

        # objs = ann[key_combine('object','sub_list')]

        # for obj in objs:
        #     obj[key_combine('class','class')] = obj[key_combine('class','other')]
        #     del obj[key_combine('class','other')]

        # class_masks = ann[key_combine('class_mask','sub_list')]

        # for class_mask in class_masks:
        #     class_mask[key_combine('class','class')] = class_mask[key_combine('class','other')]
        #     del class_mask[key_combine('class','other')]


        # ann_json = json.dumps(ann)
        # with open(ann_path, 'w') as f:
        #     f.write(ann_json)