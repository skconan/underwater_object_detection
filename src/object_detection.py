#!/usr/bin/env python
"""
    File name: object_detection.py
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
    print("pass")
    
    return return_null()    
    

if __name__=='__main__':
    rospy.init_node('object_detection_front', anonymous=False)

    seg_topic_default = "/semantic_segmentation/compressed"
    seg_topic = rospy.get_param("/object_detection/segmentation_topic", seg_topic_default)

    
    rospy.Subscriber(seg_topic, CompressedImage, image_callback)
    
    rospy.Service('object_detection_front', obj_detection_srv(), object_detection_callback)
    rospy.spin()

