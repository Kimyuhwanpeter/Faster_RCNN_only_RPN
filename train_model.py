# -*- coding: utf-8 -*-
import tensorflow as tf
import math

def ConvNets_and_RPN(input_shape=(224, 224, 3)):
    # ConvNets (CNN)
    model = tf.keras.applications.VGG16(include_top=False, input_shape=input_shape)

    h = model.get_layer("block4_conv3")
    # Region Proposal Network (RPN)
    output = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="valid", name="rpn_conv")(h.output)
    rpn_cls_output = tf.keras.layers.Conv2D(18, (1, 1), activation="sigmoid", name="rpn_cls")(output)
    rpn_reg_output = tf.keras.layers.Conv2D(36, (1, 1), activation="linear", name="rpn_reg")(output)  # linear?? ?????
    rpn_model = tf.keras.Model(inputs=model.input, outputs=[rpn_reg_output, rpn_cls_output])

    return rpn_model