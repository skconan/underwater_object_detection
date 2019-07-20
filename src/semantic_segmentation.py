#!/usr/bin/env python
"""
    File name: semantic_segmentation.py
    Author: skconan
    Date created: 2019/07/20
    Python Version: 2.7
"""
import rospy
import cv2 as cv
import numpy as np
from utilities import *
import os
from sensor_msgs.msg import CompressedImage
from keras.models import load_model


model = None
image = None



def image_callback(msg):
    global image
    arr = np.fromstring(msg.data, np.uint8)
    image = cv.imdecode(arr, 1)

def semantic_segmentation():
    global image
    r = rospy.Rate(8)

    while image is None:
        print("image is none")
        r.sleep()
        continue
    
    while not rospy.is_shutdown():
        # start = time.time()
        frame = image.copy()

        frame = cv.cvtColor(frame.copy(), cv.COLOR_BGR2RGB)
        frame = cv.resize(frame,(256,256))
        frame = frame.astype('float32')
        frame = (frame / 255.)
        pred = model.predict(frame)[0]
        
        pred = cv.resize(pred.copy(), (484,304))
        pred = pred * 255.
        pred = pred.astype('uint8')

        # print("Time",time.time()-start)
        publish_result(pred, "bgr", "semantic_segmentation")
        r.sleep()


if __name__=='__main__':
    camera_topic_default = "/vision/front/image_rect_color/compressed"
    model_file_default = "/home/skconan/model/model-color-obj-bg.hdf5"


    camera_topic = rospy.get_param(
        "/semantic_segmentation/camera_topic", camera_topic_default)
    model_file = rospy.get_param(
        "/semantic_segmentation/model_file", model_file_default)
    
    model = load_model(model_file)
    rospy.init_node('semantic_segmentation', anonymous=False)
    rospy.Subscriber(camera_topic, CompressedImage, image_callback)
    semantic_segmentation()
