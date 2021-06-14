# -*- coding: utf-8 -*-
from random import random, shuffle
from dataset import generate_anchors, generate_label, anchor_sampling
from train_model import *
from Draw_image import *

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 224,

                           "batch_size": 10,

                           "epochs": 50,

                           "num_classes": 20,

                           "lambda_val": 10.0,

                           "lr": 0.00001,
                           
                           "img_path": "D:/[1]DB/[3]detection_DB/voc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
                           
                           "txt_path": "D:/[1]DB/[3]detection_DB/voc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/xml_to_text",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint": ""})

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(FLAGS.lr, decay_steps=10000, decay_rate=0.99, staircase=True)
optim = tf.keras.optimizers.Adam(lr_schedule)

def input_func(img_path, lab_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size]) / 255.


    return img, lab_path

def read_label(file, batch_size, list_anchors, anchor_booleans):

    anchor_bool = []
    objectness_lab = []
    bbox = []
    cla = []
    for b in range(batch_size):
        f = open(tf.compat.as_bytes(file[b].numpy()), 'r')
        ground_bbox = []
        ground_class = []
        while True:
            line = f.readline()
            if not line: break
            line = line.split('\n')[0]
            
            height = int(line.split(',')[4])
            width = int(line.split(',')[5])

            xmin = float(float(line.split(',')[0]))
            xmax = float(float(line.split(',')[2]))
            ymin = float(float(line.split(',')[1]))
            ymax = float(float(line.split(',')[3]))

            xmin = float((FLAGS.img_size/width) * xmin)
            ymin = float((FLAGS.img_size/height) * ymin)
            xmax = float((FLAGS.img_size/width) * xmax)
            ymax = float((FLAGS.img_size/height) * ymax)

            generated_box_info = [xmin, ymin, xmax, ymax]       # [top-left, bottom-right]

            ground_bbox.append(generated_box_info)
            ground_class.append(int(line.split(',')[6]))

        ground_bbox = np.array(ground_bbox, dtype=np.float32)
        ground_class = np.array(ground_class, dtype=np.float32)
        anchor_boolean, objectness_label, box_regression, cla_ = generate_label(ground_class, ground_bbox, list_anchors, anchor_booleans, num_classes=FLAGS.num_classes)
        anchor_boolean = anchor_sampling(anchor_boolean, objectness_label)

        anchor_bool.append(anchor_boolean)
        objectness_lab.append(objectness_label)
        bbox.append(box_regression)
        cla.append(cla_)
        
    anchor_boolean = np.array(anchor_bool, dtype=np.float32)  
    anchor_boolean = np.reshape(anchor_boolean, [-1, 1, anchor_boolean.shape[1]]) # [batch, 1, number of anchors]
    objectness_label = np.array(objectness_lab, dtype=np.float32) # [batch, number of anchors, 2]
    box_regression = np.array(bbox, dtype=np.float32) # [batch, number of anchors, 4]
    cla = np.array(cla, dtype=np.float32) # [batch, number of anchors, 20]

    return anchor_boolean, objectness_label, box_regression, cla

def smooth_func(t):
    
    t = tf.abs(t)
    
    comparison_tensor = tf.ones((26*26*9, 4))
    smoothed = tf.where(tf.less(t, comparison_tensor), 0.5*tf.pow(t,2), t - 0.5)
    
    return smoothed

def smooth_L1(pred_box, truth_box):
    
    diff = pred_box - truth_box
    
    smoothed = tf.map_fn(smooth_func, diff)
    
    return smoothed

#@tf.function
def cal_loss(model, batch_images, anchor_boolean, objectness_label, box_regression, cla):

    with tf.GradientTape() as tape:

        reg_box_logits, rpn_cls_logits = model(batch_images, True)

        reg_box_logits = tf.reshape(reg_box_logits, [-1, 26*26*9, 4])
        rpn_cls_logits = tf.reshape(rpn_cls_logits, [-1, 26*26*9, 2])
        rpn_cls_logits = tf.nn.softmax(rpn_cls_logits, -1)

        loss_1 = tf.reduce_sum(anchor_boolean*(tf.keras.losses.categorical_crossentropy(objectness_label, rpn_cls_logits))) / tf.reduce_sum(anchor_boolean)
        loss_2 = tf.reduce_sum(objectness_label[:, :, 0] * tf.keras.losses.Huber(reduction=tf.losses.Reduction.NONE)(reg_box_logits, box_regression))
        loss_2 = loss_2 / tf.reduce_sum(objectness_label[:, :, 0])



        total_loss = loss_1 + loss_2

    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
        
    return total_loss

def main():
    
    model = ConvNets_and_RPN(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    model.summary()

    tr_txt = os.listdir(FLAGS.txt_path)
    tr_img = [FLAGS.img_path + "/" + data.split('.')[0] + ".jpg" for data in tr_txt]
    tr_lab = [FLAGS.txt_path + "/" + data for data in tr_txt]

    list_anchors, anchor_booleans = generate_anchors()

    count = 0
    for epoch in range(FLAGS.epochs):

        tr_generator = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
        tr_generator = tr_generator.shuffle(len(tr_img))
        tr_generator = tr_generator.map(input_func)
        tr_generator = tr_generator.batch(FLAGS.batch_size)
        tr_generator = tr_generator.prefetch(tf.data.experimental.AUTOTUNE)

        tr_iter = iter(tr_generator)
        tr_idx = len(tr_img) // FLAGS.batch_size

        for step in range(tr_idx):
            batch_images, batch_labels = next(tr_iter)
            anchor_boolean, objectness_label, box_regression, cla = read_label(batch_labels, FLAGS.batch_size, list_anchors, anchor_booleans)

            loss = cal_loss(model, batch_images, anchor_boolean, objectness_label, box_regression, cla)

            if count % 10 == 0:
                print("<Faster RCNN> Epoch: [{}/{}], Iteration: {}, Loss = {}".format(epoch, FLAGS.epochs, count + 1, loss))
            
            if count % 100 == 0:
                reg_box_logits, rpn_cls_logits = model(batch_images, False)

                reg_box_logits = tf.reshape(reg_box_logits, [-1, 26*26*9, 4])
                rpn_cls_logits = tf.reshape(rpn_cls_logits, [-1, 26*26*9, 2])

                reg_box_logit = reg_box_logits[0]
                rpn_cls_logit = rpn_cls_logits[0, :, 0]
                cla_ = cla[0]

                anchors = np.array(list_anchors, dtype=np.float32)
                bboxes = get_box_from_delta(anchors, reg_box_logit)
                selected_indices = tf.image.non_max_suppression(bboxes,
                                                                rpn_cls_logit,
                                                                max_output_size=200,
                                                                score_threshold=0.5)
                selected_boxes = tf.gather(bboxes, selected_indices)

                # 이건 RPN만 학습시킨것

                print(4)


            count += 1


if __name__ == "__main__":
    main()
