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
from sklearn.externals import joblib
from zeabus_utility.srv import VisionBuoy,VisionBuoyResponse

model = joblib.load("/home/zeabus/catkin_ws/src/underwater_object_detection/src/forest_model.sav")
image = None 
mask_tmp = np.zeros((rows,cols),np.uint8)


def buoy_callback(msg):
    return object_detection("buoy")

def return_null():
    print("return null")
    msg = obj_detection_msg()
    msg.appear = False

    img_msg = CompressedImage()
    img_msg.header.stamp = rospy.Time.now()
    msg.mask = img_msg
    return msg

# def return_result(img):
#     msg = obj_detection_msg()
#     msg.appear = True

#     img_msg = CompressedImage()
#     img_msg.header.stamp = rospy.Time.now()
#     img_msg.format = "jpeg"
#     img_msg.data = np.array(cv.imencode('.jpg', img)[1]).tostring()
#     msg.mask = img_msg
#     return msg

def return_null_buoy():
    res = VisionBuoyResponse()
    res.found = 0
    res.cx = 0
    res.cy = 0
    res.area = 0
    res.score = 0
    return res

def return_result_buoy(x,y,area,score):
    res = VisionBuoyResponse()
    res.found = 1
    res.cx = x
    res.cy = y
    res.area = area
    res.score = score
    return res

# def object_detection_callback(msg):
#     obj_name = msg.obj
#     print(obj_name)
#     return object_detection(obj_name)

def image_callback(msg):
    global image
    arr = np.fromstring(msg.data, np.uint8)
    image = cv.imdecode(arr, 1)

def object_prediction(mask):
    global model
    mask = cv.resize(mask,(242,152))
    onehot = np.reshape(mask.copy(), [1, 242 * 152])
    output = model.predict(onehot)
    print(output)
    y = output[0]
    print("Y",y)
    return y

def object_detection(obj_name):
    global image, model, mask_tmp
    
    if image is None:
        print("image is None")
        return return_null()
    area_min = 2000

    lower = rospy.get_param("/object_detection/object_color_range/"+ obj_name +"/lower")
    upper = rospy.get_param("/object_detection/object_color_range/"+ obj_name +"/upper")

    lower = range_str2list(lower)
    upper = range_str2list(upper)
    
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower = np.array(lower,np.uint8)
    upper = np.array(upper,np.uint8)
    mask = cv.inRange(hsv, lower, upper)

    _, mask = cv.threshold(mask, 100, 255, cv.THRESH_BINARY)
    # mask = cv.erode(mask,get_kernel())
    # mask = cv.dilate(mask,get_kernel())

    publish_result(mask,"object_detection/mask")

    pose = []
    
    if int((cv.__version__).split(".")[0]) < 4:
        _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    rows,cols = mask.shape
    result = image.copy()

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < area_min:
            continue
        print(area)
        x, y, w, h = np.int0(cv.boundingRect(cnt))
        
        mask_pred = mask_tmp.copy()
        cv.rectangle(mask_pred,(x,y),(x+w,y+h),(255,255,255),-1)
        mask_pred = cv.bitwise_and(mask_pred,mask)
        mission_name = object_prediction(mask_pred)

        if mission_name == "gate":
            # red
            cv.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
        elif mission_name == "buoy":
            # blue
            cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
            found = True
            x = x+w/2.
            y = y+h/2.
            x = (x - cols/2.) / (cols/2.)
            y = -(y - rows/2.) / (rows/2.)
            area = (w*h)*1.0 / (rows*cols)
            score = 20
            print("Found:",x,y,area)
            publish_result(image,"object_detection/result")
            return return_result_buoy(x,y,area,score)
            
    return return_null_buoy()
    
    

if __name__=='__main__':
    rospy.init_node('object_detection_front', anonymous=False)

    seg_topic_default = "/semantic_segmentation/compressed"
    seg_topic = rospy.get_param("/object_detection/segmentation_topic", seg_topic_default)

    
    rospy.Subscriber(seg_topic, CompressedImage, image_callback)
    
    # rospy.Service('object_detection_front', obj_detection_srv(), object_detection_callback)
    rospy.Service('/vision/buoy', VisionBuoy(), buoy_callback)
    rospy.spin()
    # while not rospy.is_shutdown():
    #     object_detection("buoy")

