#!/usr/bin/python
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

model_file = "/home/zeabus/catkin_ws/src/vision_ml/src/model-color-obj.hdf5"
model = load_model(model_file)
image = None



def image_callback(msg):
    global image
    arr = np.fromstring(msg.data, np.uint8)
    image = cv.imdecode(arr, 1)

def semantic_segmentation():
    global image
    r = rospy.Rate(10)

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
    topic_name = "/vision/front/image_rect_color/compressed"
    rospy.init_node('semantic_segmentation')
    rospy.Subscriber(topic_name, CompressedImage, image_callback)
    semantic_segmentation()
