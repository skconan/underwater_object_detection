#!/usr/bin/python
"""
    File name: gate_detection.py
    Author: skconan
    Date created: 2019/04/13
    Python Version: 3.6
"""
import rospy
import cv2 as cv
import numpy as np
import tensorflow as tf
# from bg_subtraction import bg_subtraction
from semantic_segmentation import Pix2Pix
from utilities import *
from operator import itemgetter
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# model_file = "./pix2pix_checkpoints_seg_obj_color/60-ckpt-4"
model_file = "./1180-ckpt-60"
pix2pix = Pix2Pix(postfix='',train=False)
pix2pix.checkpoint.restore(model_file)
image = None

def publish_result(img, type, topicName):
    """
        publish picture
    """
    if img is None:
        img = np.zeros((200, 200))
        type = "gray"
    bridge = CvBridge()
    pub = rospy.Publisher(
        str(topicName), Image, queue_size=10)
    if type == 'gray':
        msg = bridge.cv2_to_imgmsg(img, "mono8")
    elif type == 'bgr':
        msg = bridge.cv2_to_imgmsg(img, "bgr8")
    pub.publish(msg)


def img_callback(msg):
    global image
    arr = np.fromstring(msg.data, np.uint8)
    image = cv.imdecode(arr, 1)
    image = cv.resize(image, None, fx=0.5, fy=0.5)
    # hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

def segmentation_pix2pix(img):
    # with tf.device('/device:GPU:0'):
    #     img = tf.convert_to_tensor(np.float32(img))
    #     img = resize(img, 256, 256)
    #     img, result = pix2pix.predict_images(img)
    img = tf.convert_to_tensor(np.float32(img))
    img = resize(img, 256, 256)
    img, result = pix2pix.predict_images(img)
    # hsv = cv.cvtColor(result, cv.COLOR_BGR2HSV)
    # cv.imshow("result_tf_color",result.copy())
    # lower = np.array([22, 200, 200], np.uint8)
    # upper = np.array([38, 255, 255], np.uint8)
    # mask = cv.inRange(hsv, lower, upper)
    return result


def find_object(mask):
    area_min = 500
    pose = []
    _, th = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    
    if int((cv.__version__).split(".")[0]) < 4:
        _, contours, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < area_min:
            continue
        x, y, w, h = np.int0(cv.boundingRect(cnt))
        # img = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        pose.append([x, y, area])
        # rect = cv.minAreaRect(cnt)
        # box = cv.boxPoints(rect)
        # box = np.int0(box)
        # (x,y),(w,h), angle = box
        # im = cv.drawContours(im,[box],0,(0,0,255),2)

    if len(pose) > 0:
        pose = sorted(pose, key=itemgetter(2), reverse=True)
        pose = pose[0]
    return pose


def segmentation(img):
    font = cv.FONT_HERSHEY_SIMPLEX
    result = img.copy()
    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # lower = {
    #     'pink': np.array([145, 200, 200], np.uint8),
    #     'yellow': np.array([22, 200, 200], np.uint8),
    #     'red': np.array([160, 200, 200], np.uint8),
    #     'green': np.array([50, 200, 200], np.uint8),
    #     'cyan': np.array([82, 200, 200], np.uint8),
    # }
    # upper = {
    #     'pink': np.array([155, 255, 255], np.uint8),
    #     'yellow': np.array([38, 255, 255], np.uint8),
    #     'red': np.array([179, 255, 255], np.uint8),
    #     'green': np.array([65, 255, 255], np.uint8),
    #     'cyan': np.array([97, 255, 255], np.uint8),
    # }

    lower = {
        'pink': np.array([255, 0, 255], np.uint8),
        'yellow': np.array([0, 255, 255], np.uint8),
        'red': np.array([0, 0, 255], np.uint8),
        'green': np.array([0, 255, 0], np.uint8),
        'pupple': np.array([250, 0, 120], np.uint8),
        'cyan': np.array([255, 255, 0], np.uint8),
    }
    upper = {
        'pink': np.array([155, 255, 255], np.uint8),
        'yellow': np.array([38, 255, 255], np.uint8),
        'red': np.array([179, 255, 255], np.uint8),
        'green': np.array([65, 255, 255], np.uint8),
        'pupple': np.array([255, 0, 127], np.uint8),
        'cyan': np.array([97, 255, 255], np.uint8),
    }

    color = ['red', 'yellow', 'pupple']
    obj = {'red': 'gate',
           'yellow': 'slot', 'pupple': 'dice'}

    for c in color:
        # mask = cv.inRange(hsv,lower[c],upper[c])
        mask = cv.inRange(img, lower[c], lower[c])
        res = find_object(mask)
        if len(res) > 0:

            # for p in pose_list:
            x, y, _ = res
            if x > 484//2:
                x -= 50
            else:
                x += 50
            if y > 304//2:
                y -= 50
            else:
                y += 50
            cv.putText(result, obj[c], (x, y), font,
                       1, (0, 0, 0), 2, cv.LINE_AA)
    return result


def main():
    global image

    while not rospy.is_shutdown():
        # ret, frame = cap.read()
        if image is None:
            print("image is none")
            continue

        if not ret:
            print_debug("Cannot read frame")
            break
        frame = cv.resize(frame, (484, 304))

        frame_tf = cv.cvtColor(frame.copy(), cv.COLOR_BGR2RGB)

        result_tf = segmentation_pix2pix(frame_tf)
        # result = segmentation(result_tf)

        publish_result(result_tf, "bgr", "semantic_segmentation")

        # k = cv.waitKey(1) & 0xff
        # if k == ord('q'):
        #     break



if __name__=='__main__':
    rospy.init_node('object_detection')
    main()
