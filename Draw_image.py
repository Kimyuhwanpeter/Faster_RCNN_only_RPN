# -*- coding:utf-8 -*-
from PIL import ImageDraw, Image
import numpy as np

def get_box_from_delta(anchors, boxes):

    all_anc_ctr_x = anchors[..., 0]
    all_anc_ctr_y = anchors[..., 1]
    all_anc_width = anchors[..., 2]
    all_anc_height = anchors[..., 3]

    all_bbox_width = np.exp(boxes[..., 2]) * all_anc_width
    all_bbox_height = np.exp(boxes[..., 3]) * all_anc_height
    all_bbox_ctr_x = (boxes[..., 0] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (boxes[..., 1] * all_anc_height) + all_anc_ctr_y

    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x2 = all_bbox_width + x1
    y2 = all_bbox_height + y1
    

    return np.stack([y1, x1, y2, x2], axis=-1)