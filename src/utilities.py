#!/usr/bin/python
"""
    File name: utilities.py
    Author: skconan
    Date created: 2019/04/09
    Python Version: 3.6
"""

import os
import numpy as np
import tensorflow as tf
import colorama
from PIL import Image
import cv2 as cv
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
import rospy

colorama.init()
DEBUG = True


def print_debug(*args, **kwargs):
    global DEBUG
    text = ""
    if not "mode" in kwargs:
        mode = "DETAIL"
    else:
        mode = kwargs['mode']
    color_mode = {
        "METHOD": colorama.Fore.BLUE,
        "RETURN": colorama.Fore.GREEN,
        "DETAIL": colorama.Fore.YELLOW,
        "DEBUG": colorama.Fore.RED,
        "END": colorama.Style.RESET_ALL,
    }
    if DEBUG:
        for t in args:
            text += " "+str(t)
        print(color_mode[mode] + text + color_mode["END"])


def get_file_path(dir_name):
    file_list = os.listdir(dir_name)
    files = []
    for f in file_list:
        abs_path = os.path.join(dir_name, f)
        if os.path.isdir(abs_path):
            files = files + get_file_path(abs_path)
        else:
            files.append(abs_path)

    return files


def get_file_name(img_path):
    if "\\" in img_path:
        name = img_path.split('\\')[-1]
    else:
        name = img_path.split('/')[-1]

    name = name.replace('.gif', '')
    name = name.replace('.png', '')
    name = name.replace('.jpg', '')
    return name


def normalize(img_float):
    result = img_float / 255.
    return result


def get_kernel(shape='rect', ksize=(3, 3)):
    if shape == 'rect':
        return cv.getStructuringElement(cv.MORPH_RECT, ksize)
    elif shape == 'ellipse':
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize)
    elif shape == 'plus':
        return cv.getStructuringElement(cv.MORPH_CROSS, ksize)
    else:
        return None


def publish_result(img, type, topic_name):
    """
        publish picture
    """
    #### Create CompressedIamge ####
    if img is None:
        img = np.zeros((200, 200))
        type = "gray"

    pub = rospy.Publisher(
        str(topic_name), CompressedImage, queue_size=10)
    
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv.imencode('.jpg', img)[1]).tostring()

    pub.publish(msg)
