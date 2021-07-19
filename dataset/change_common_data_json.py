from ymlib.common_dataset_api import key_combine
import os
import cv2 as cv
import json
import glob
import tqdm

def test(dataset_dir):
   for i0, filename in enumerate(glob.glob(os.path.join(dataset_dir, '*'))):
        print(filename)
        img = cv.imread(filename)
        h, w = img.shape[:2]
        min_ = min(1000/h, 1000/w)
        img_resize = cv.resize(img, (0, 0), fx=min_, fy=min_)
        cv.imwrite(os.path.join(dataset_dir, str(i0).zfill(5)+'.jpg'), img_resize)
    
    
        ann_paths = glob.glob(os.path.join(dataset_dir, 'data', '*.json'))

        for ann_path in tqdm.tqdm(ann_paths):
            with open(ann_path) as f:
                ann = json.load(f)

            ann[key_combine('class','class')] = ann[key_combine('class','other')]
            del ann[key_combine('class','other')]

            objs = ann[key_combine('object','sub_list')]

            for obj in objs:
                obj[key_combine('class','class')] = obj[key_combine('class','other')]
                del obj[key_combine('class','other')]

            class_masks = ann[key_combine('class_mask','sub_list')]

            for class_mask in class_masks:
                class_mask[key_combine('class','class')] = class_mask[key_combine('class','other')]
                del class_mask[key_combine('class','other')]

            ann_json = json.dumps(ann)
            with open(ann_path, 'w') as f:
                f.write(ann_json)


if __name__ == '__main__':
    # dataset_dir = '/Users/yanmiao/yanmiao/data/hun_sha_di_pian/origin'
    dataset_dir = '/Users/yanmiao/yanmiao/data-common/supervisely'
    test(dataset_dir)
    # save_dir = '/Users/yanmiao/yanmiao/data/hun_sha_di_pian/downSample1000'

 
