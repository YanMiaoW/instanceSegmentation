from dataset.common_dataset_api import BODY_PART_CHOICES, key_combine, key_decompose
import cv2 as cv
import numpy as np
import math


def mask2box(mask):
    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    bound_boxs_xywh = [cv.boundingRect(contour) for contour in contours]

    x1s = [x for x, y, w, h in bound_boxs_xywh]
    y1s = [y for x, y, w, h in bound_boxs_xywh]
    x2s = [x+w for x, y, w, h in bound_boxs_xywh]
    y2s = [y+h for x, y, w, h in bound_boxs_xywh]

    if not all(len(s) > 0 for s in [x1s, y1s, x2s, y2s]):
        return None
    else:
        return [min(x1s), min(y1s), max(x2s), max(y2s)]


def draw_box(image, box: list, color=(0, 255, 0), thickness=None):
    if thickness is None:
        thickness = int(max(min(image.shape[:2])*0.005, 1))
    cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)


def draw_mask(image, mask, color=(255, 255, 0), alpha=0.5):
    select = mask > 127
    h, w = mask.shape
    select3 = select[:, :, np.newaxis].repeat(3, axis=2)
    color_block = np.array(color, dtype=np.uint8)[
        np.newaxis, np.newaxis, :].repeat(h, axis=0).repeat(w, axis=1)
    image[select3] = image[select3]*(1-alpha) + color_block[select3]*alpha


def draw_label(image, text: str, xy: list, color=[0, 255, 0], size_percent=1, thickness=1):
    h, w = image.shape[:2]
    r = math.hypot(h, w)
    r *= 0.001

    x, y = xy
    x += int(r*10)
    y += int(r*30)

    cv.putText(image, text=text, org=[x, y],
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=r*size_percent, color=color,
               thickness=thickness, lineType=cv.LINE_AA)


def draw_keypoint(image, keypoint: dict, labeled=False):

    part_names = ["right_shoulder", "right_elbow", "right_wrist",
                  "left_shoulder", "left_elbow", "left_wrist",
                  "right_hip", "right_knee", "right_ankle",
                  "left_hip", "left_knee", "left_ankle",
                  "head", "neck", 'right_ear', 'left_ear',
                  'nose', 'right_eye', 'left_eye']

    connection = [[16, 19], [13, 17], [4, 5],
                  [19, 17], [17, 14], [5, 6],
                  [17, 18], [14, 4], [1, 2],
                  [18, 15], [14, 1], [2, 3],
                  [4, 10], [1, 7], [10, 7],
                  [10, 11], [7, 8], [11, 12],
                  [8, 9], [16, 4], [15, 1]]

    connection2key = {i+1: key for i, key in enumerate(part_names)}

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
              [255, 255, 0], [170, 255, 0], [85, 255, 0],
              [0, 255, 0], [0, 255, 85], [0, 255, 170],
              [0, 255, 255], [0, 170, 255], [0, 85, 255],
              [0, 0, 255], [85, 0, 255], [170, 0, 255],
              [255, 0, 255], [255, 0, 170], [255, 0, 85], [34, 177, 60]]

    key2color = {key: np.array(color, dtype=np.int64)
                 for key, color in zip(part_names, colors)}

    Rfactor = min(image.shape[:2])*0.1
    Rpoint = int(min(10, max(Rfactor*10, 4)))
    Rline = int(min(10, max(Rfactor*5, 2))*0.5)

    def get_x_y_status(key):
        partname_key_type = key_combine(key, 'sub_dict')
        if partname_key_type in keypoint:
            status = keypoint[partname_key_type][key_combine(
                'status', 'keypoint_status')]
            xy = keypoint[partname_key_type][key_combine('point', 'point_xy')]
            x, y = xy
            return x, y, status
        else:
            return 0, 0, 'missing'

    for idx in range(len(connection)):
        idx1, idx2 = connection[idx]
        key1, key2 = connection2key[idx1], connection2key[idx2]
        x1, y1, v1 = get_x_y_status(key1)
        x2, y2, v2 = get_x_y_status(key2)

        if v1 == 'missing' or v2 == 'missing':
            continue

        cx = (x1+x2)/2
        cy = (y1+y2)/2
        length = math.hypot(x1-x2, y1-y2)
        angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
        draw = image.copy()
        color = key2color[key1]*0.5+key2color[key2]*0.5
        cv.ellipse(draw, (int(cx), int(cy)), (int(length/2), Rline),
                   int(angle), 0, 360, color, -1)
        image[:] = image*0.5 + draw*0.5

    for partname_key_type in keypoint:
        key, _ = key_decompose(partname_key_type)

        x, y, status = get_x_y_status(key)
        x, y = int(x), int(y)
        color = key2color[key].tolist()

        if status == 'vis':
            cv.circle(image, (x, y), Rpoint, color, thickness=-1)

        elif status == 'not_vis':
            cv.rectangle(image, (x-Rpoint-1, y-Rpoint-1), (x+Rpoint+1, y+Rpoint+1),
                         color, int(Rline*0.4))

        if status != 'missing' and labeled:
            draw_label(image, key, [x, y], color, size_percent=0.5)

