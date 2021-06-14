# -*- coding:utf-8 -*-
import numpy as np
import math

anchor_ratios = [[1,1],[1/math.sqrt(2),math.sqrt(2)],[math.sqrt(2),1/math.sqrt(2)]]
anchor_scales = [32, 64, 128]

anchor_count = len(anchor_ratios) * len(anchor_scales)

# subsampled_ratio=8 --> ROI pooing 3 times
def generate_anchors(rpn_kernel_size=3, subsampled_ratio=8,
                     anchor_sizes=[32, 64, 128], anchor_aspect_ratio=anchor_ratios):
    '''
    Input : subsample_ratio (=Pooled ratio)
    generate anchor in feature map (important!!!). Then project it to original image.
    Output : list of anchors (x,y,w,h) and anchor_boolean (ignore anchor if value equals 0)
    '''

    list_anchors = []
    anchor_booleans = []

    start_center = int(rpn_kernel_size / 2)
    anchor_center = [start_center - 1, start_center]

    subsampled_height = 224/subsampled_ratio       # = 28
    subsampled_width = 224/subsampled_ratio         # = 28

    while (anchor_center != [subsampled_width - (1 + start_center),
                             subsampled_height - (1 + start_center)]):

        anchor_center[0] += 1 #Increment x-axis

        #If sliding window reached last center, increase y-axis
        if anchor_center[0] > subsampled_width - (1 + start_center):
            anchor_center[1] += 1
            anchor_center[0] = start_center

        #anchors are referenced to the original image. 
        #Therefore, multiply downsampling ratio to obtain input image's center 
        anchor_center_on_image = [anchor_center[0]*subsampled_ratio, anchor_center[1]*subsampled_ratio]

        for size in anchor_sizes:
            
            for a_ratio in anchor_aspect_ratio:
                # [x,y,w,h]
                anchor_info = [anchor_center_on_image[0], anchor_center_on_image[1], size*a_ratio[0], size*a_ratio[1]]

                # check whether anchor crosses the boundary of the image or not
                if (anchor_info[0] - anchor_info[2]/2 < 0 or anchor_info[0] + anchor_info[2]/2 > 224 or 
                                        anchor_info[1] - anchor_info[3]/2 < 0 or anchor_info[1] + anchor_info[3]/2 > 224) :
                    anchor_booleans.append([0.0])       # if anchor crosses boundary, anchor_booleans=0

                else:
                    anchor_booleans.append([1.0])

                list_anchors.append(anchor_info)
    
    return list_anchors, anchor_booleans    # 26 * 26 * 9 만큼의 앵커박스를 만들고 여기서 범위에 맞는것만 쓴다는 매커니즘이다

def generate_label(class_labels, ground_boxes, anchors, anchor_boolens, num_classes=20,
                   neg_threshold=0.3, pos_threshold=0.7):
    '''
    Input  : classes, ground truth box (top-left, bottom-right), all of anchors, anchor booleans.
    Compute IoU to get positive, negative samples.
    if IoU > 0.7, positive / IoU < 0.3, negative / Otherwise, ignore
    Output : anchor booleans (to know which anchor to ignore), objectness label, regression coordinate in one image
    '''
    number_of_anchors = len(anchors)

    anchor_boolean_array = np.reshape(np.asarray(anchor_booleans),(number_of_anchors, 1))

    # IoU is more than threshold or not.
    objectness_label_array = np.zeros((number_of_anchors, 2), dtype=np.float32)

    # delta(x, y, w, h)
    box_regression_array = np.zeros((number_of_anchors, 4), dtype=np.float32)

    # belongs to which object for every anchor
    class_array = np.zeros((number_of_anchors, num_classes), dtype=np.float32)

    for j in range(ground_boxes.shape[0]):

        gt_box_xmin = ground_boxes[j][0]
        gt_box_ymin = ground_boxes[j][1]
        gt_box_xmax = ground_boxes[j][2]
        gt_box_ymax = ground_boxes[j][3]

        gt_box_area = (gt_box_xmax - gt_box_xmin + 1) * (gt_box_ymax - gt_box_ymin + 1)

        for i in range(number_of_anchors):
            # IOU
            if int(anchor_boolens[i][0]) == 0:
                continue
            anchor = anchors[i] # [x, y, w, h]

            anchor_box_xmin = anchor[0] - anchor[2] / 2
            anchor_box_ymin = anchor[1] - anchor[3] / 2
            anchor_box_xmax = anchor[0] + anchor[2] / 2
            anchor_box_ymax = anchor[1] + anchor[3] / 2

            anchor_box_area = (anchor_box_xmax - anchor_box_xmin + 1) * (anchor_box_ymax - anchor_box_ymin + 1)

            int_rect_xmin = max(gt_box_xmin, anchor_box_xmin)
            int_rect_ymin = max(gt_box_ymin, anchor_box_ymin)
            int_rect_xmax = min(gt_box_xmax, anchor_box_xmax)
            int_rect_ymax = min(gt_box_ymax, anchor_box_ymax)

            # if the boxes do not intersect, difference = 0
            int_rect_area = max(0, int_rect_xmax - int_rect_xmin + 1)*max(0, int_rect_ymax - int_rect_ymin)

            IOU = float(int_rect_area / (gt_box_area + anchor_box_area - int_rect_area))

            if IOU >= pos_threshold:
                objectness_label_array[i][0] = 1.0
                objectness_label_array[i][1] = 0.0

                class_label = class_labels[j]
                class_array[i][int(class_label)] = 1.0  # one hot encode

                gt_box_x = ground_boxes[j][0] + ground_boxes[j][2] / 2
                gt_box_y = ground_boxes[j][1] + ground_boxes[j][3] / 2
                gt_box_w = ground_boxes[j][2] - ground_boxes[j][0]
                gt_box_h = ground_boxes[j][3] - ground_boxes[j][1]

                # addaptive to loss
                delta_x = (gt_box_x - anchor[0])/anchor[2]
                delta_y = (gt_box_y - anchor[1])/anchor[3]
                delta_w = math.log(gt_box_w/anchor[2])
                delta_h = math.log(gt_box_h/anchor[3])

                box_regression_array[i][0] = delta_x
                box_regression_array[i][1] = delta_y
                box_regression_array[i][2] = delta_w
                box_regression_array[i][3] = delta_h

            if IOU <= neg_threshold:
                if int(objectness_label_array[i][0]) == 0:
                    objectness_label_array[i][1] = 1.0
            if IOU > neg_threshold and IOU < pos_threshold:
                if int(objectness_label_array[i][0]) == 0 and int(objectness_label_array[i][1]) == 0:
                    anchor_boolean_array[i][0] = 0.0

    return anchor_boolean_array, objectness_label_array, box_regression_array, class_array

def anchor_sampling(anchor_booleans, objectness_label, anchor_sampling_amount=128):

    '''
    Input : anchor booleans and objectness label
    fixed amount of negative anchors and positive anchors for training. 
    If we use all the neg and pos anchors, model will overfit on the negative samples.
    Output: Updated anchor booleans. 
    '''    
    # 이게 효과가 정확히 어떤것인지 아직은 정확하게 이해가 안간다
    positive_count = 0
    negative_count = 0
    
    for i in range(objectness_label.shape[0]):
        if int(objectness_label[i][0]) == 1: #If the anchor is positive

            if positive_count > anchor_sampling_amount: #If the positive anchors are more than the threshold amount, set the anchor boolean to 0.

                anchor_booleans[i][0] = 0.0

            positive_count += 1

        if int(objectness_label[i][1]) == 1: #If the anchor is negatively labelled.
            if negative_count > anchor_sampling_amount: #If the negative anchors are more than the threshold amount, set the boolean to 0.

                anchor_booleans[i][0] = 0.0

            negative_count += 1

    return anchor_booleans

list_anchors, anchor_booleans = generate_anchors()
print(len(anchor_booleans)) 