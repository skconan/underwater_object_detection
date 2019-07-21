#!/usr/bin/env python
"""
    File name: object_detection_front.py
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
from object_detection.srv import obj_detection_srv
from object_detection.msg import obj_detection_msg
from operator import itemgetter

image = None

def return_null():
    msg = obj_detection_msg()
    msg.appear = False

    img_msg = CompressedImage()
    img_msg.header.stamp = rospy.Time.now()
    msg.mask = img_msg
    return msg

def return_result(img):
    msg = obj_detection_msg()
    msg.appear = True

    img_msg = CompressedImage()
    img_msg.header.stamp = rospy.Time.now()
    img_msg.format = "jpeg"
    img_msg.data = np.array(cv.imencode('.jpg', img)[1]).tostring()
    msg.mask = img_msg

def object_detection_callback(msg):
    obj_name = msg.obj
    return object_detection(obj_name)

def image_callback(msg):
    global image
    arr = np.fromstring(msg.data, np.uint8)
    image = cv.imdecode(arr, 1)

def object_detection(obj_name):
    global image, model
    
    area_min = 500

    lower = rospy.get_param("/object_detection/object_color_range/"+ obj_name +"/lower")
    upper = rospy.get_param("/object_detection/object_color_range/"+ obj_name +"/upper")

    lower = range_str2list(lower)
    upper = range_str2list(upper)
    print(lower,upper)
    
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower = np.array(lower,np.uint8)
    upper = np.array(upper,np.uint8)
    mask = cv.inRange(hsv, lower, upper)

    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    mask = cv.erode(mask,get_kernel())
    mask = cv.dilate(mask,get_kernel())

    pose = []
    
    if int((cv.__version__).split(".")[0]) < 4:
        _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < area_min:
            continue
        x, y, w, h = np.int0(cv.boundingRect(cnt))
        pose.append([x, y, area])

    if len(pose) > 0:
        # pose = sorted(pose, key=itemgetter(2), reverse=True)
        # pose = pose[0]
        return return_result(mask)
    else:
        return return_null()    
    

if __name__=='__main__':
    rospy.init_node('object_detection_front', anonymous=False)

    seg_topic_default = "/semantic_segmentation/compressed"
    seg_topic = rospy.get_param("/object_detection/segmentation_topic", seg_topic_default)

    
    rospy.Subscriber(seg_topic, CompressedImage, image_callback)
    
    rospy.Service('object_detection_front', obj_detection_srv(), object_detection_callback)
    rospy.spin()

