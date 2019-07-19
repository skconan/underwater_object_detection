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
from sensor_msgs.msg import CompressedImage
from zeabus_utility.msg import HeaderFloat64

from utilities import *
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import math 

image = None
pressure = 0 
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 3, 1.0)

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


def normalize(gray):
    return np.uint8(255 * (gray - gray.min()) / (gray.max() - gray.min()))

def kmean(img, k):
    global criteria
    Z = img.reshape((-1,1))

    # convert to np.float32
    Z = np.float32(Z)

    
    K = k
    ret,label,center=cv.kmeans(Z,K,None,criteria,3,cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    return res2

def image_callback(msg):
    global image
    print("image_callback")
    arr = np.fromstring(msg.data, np.uint8)
    image = cv.imdecode(arr, 1)
    image = cv.resize(image, None, fx=0.25, fy=0.25)


# def pressure_callback(msg):
#     global pressure
#     print("pressure_callback")
#     pressure = msg.data

def bg_subtraction(img):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    gray,_,_ = cv.split(hsv)
    # print("z",z)
    # z = max(z,0)
    # bg_kernel_size = 30 + z*5
    # fg_kernel_size = bg_kernel_size * 1.5 
    
    # bg_kernel_size = int(bg_kernel_size)
    # fg_kernel_size = int(fg_kernel_size)

    # if bg_kernel_size% 2 == 0:
    #     bg_kernel_size += 1
    
    # if fg_kernel_size% 2 == 0:
    #     fg_kernel_size += 1

    # print(bg_kernel_size)    
    # bg = cv.medianBlur(gray, bg_kernel_size*2 + 1)
    start = time.time()
    bg = kmean(gray, k=1)
    fg = kmean(gray, k=3)
    # bg = cv.blur(gray, (bg_kernel_size,bg_kernel_size))
    # fg = cv.blur(gray, (5,5))
    # fg = cv.medianBlur(gray, fg_kernel_size)

    sub_sign = np.int16(fg) - np.int16(bg)
    # sub_pos = np.clip(sub_sign.copy(),0,sub_sign.copy().max())
    sub_neg = np.clip(sub_sign.copy(),sub_sign.copy().min(),0)
    # sub_pos = normalize(sub_pos)
    sub_neg = normalize(sub_neg)
    
    _, obj_neg = cv.threshold(
        sub_neg, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )

    # _, obj_pos = cv.threshold(
    #     sub_pos, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    # )
    
    # bg = np.uint8(bg)
    # fg = np.uint8(fg)
    
    print(time.time() - start)
    # publish_result(bg, "gray", "/path_bg")
    # publish_result(fg, "gray", "/path_fg")
    publish_result(obj_neg, "gray", "/path_obj_neg")
    # publish_result(obj_pos, "gray", "/path_obj_pos")
    return obj_neg

def find_triangle(mask):
    _, mask = cv.threshold(mask, 20, 255, cv.THRESH_BINARY)
    hierarchy, contours, _ = cv.findContours(
        mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    count = 0
    res = cv.merge((mask, mask, mask))
    cv.drawContours(res, contours, -1, (0, 255, 255), 1)

    for cnt in contours:
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.01 * peri, True)
        # cv.drawContours(res,approx,-1,(255,0,0),1)
        if len(approx) == 3:
            count += 1
            cv.drawContours(res, cnt, -1, (0, 0, 255), 2)
    # cv.imshow('mask2', res)
    # cv.imshow('mask3', mask)
    publish_result(res, "bgr", "/path_tri")

    # cv.waitKey(1)
    return count


def is_full_path(mask):
    tmp = mask.copy()
    tmp.fill(0)
    _, mask = cv.threshold(mask, 20, 255, cv.THRESH_BINARY)
    hierarchy, contours, _ = cv.findContours(
        mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(tmp, contours, -1, (255, 255, 255), 1)
    count = find_triangle(tmp)
    print(count)
    # cv.imshow('mask1', mask)
    # cv.imshow('res1', tmp)
    # cv.waitKey(1)
    if 2 <= count <= 3:
        return True
    return False



def find_path():
    global image

    
    area_ratio_upper = 0.6
    area_ratio_lower = 0.35

    while not rospy.is_shutdown():
        img = image
        if img is None:
            continue
        is_path = False
        cx, cy = -1, -1
        res = img.copy()
        # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        mask = bg_subtraction(img)
        # mask = cv.inRange(hsv, lowerb, upperb)
        # erode = cv.erode(mask, get_kernel('rect', (3, 3)))
        # dilate = cv.dilate(erode, get_kernel("rect", (7, 7)))
        # _, dilate = cv.threshold(dilate, 20, 255, cv.THRESH_BINARY)

        hierarchy, contours, _ = cv.findContours(
            mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

        for cnt in contours:
            type_path = 'Not'
            rect = cv.minAreaRect(cnt)
            (x, y), (width, height), angle = rect
            area_cnt = cv.contourArea(cnt)
            if area_cnt <= 1000:
                x, y = int(x), int(y)
                width, height = int(width), int(height)
                cv.rectangle(mask, (x - int(width / 2), y - int(height / 2)),
                             (x + width, y + height), (0, 0, 0), -1)
                continue
            area_box = width * height
            area_ratio = area_cnt / area_box

            xx, yy, ww, hh = cv.boundingRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(res, [box], 0, (0, 0, 255), 1)

            if area_ratio_lower <= area_ratio <= area_ratio_upper:
                roi = mask.copy()
                roi.fill(0)
                roi1 = roi.copy()
                cv.drawContours(roi, [box], 0, (255, 255, 255), -1)
                cv.drawContours(roi1, [box], 0, (255, 255, 255), -1)
                roi = roi & mask
                roi = roi ^ roi1
                roi[:5, :] = 0
                roi[-5:, :] = 0
                roi = cv.erode(roi, get_kernel('rect', (5, 5)))
                is_path = is_full_path(roi)
                if is_path:
                    angle -= 22.5
                    cx = x
                    cy = y
                    rect = (x, y), (2, int(height / 2.0)), angle
                    box = cv.boxPoints(rect)
                    box = np.int0(box)
                    cv.drawContours(res, [box], 0, (0, 255, 255), 2)
                    cv.circle(res, (int(x), int(y)), 3, (0, 255, 0), -1)
                    type_path = 'Full'

            elif area_ratio > area_ratio_upper + 0.1:
                # path_w = 30
                # 30/2 = 15
                # path_h = 6
                wh_ratio_lower = 15/6.0
                wh_ratio_upper = 30/6.0
                width = float(width)
                if wh_ratio_lower <= max(width/height, height/width) <= wh_ratio_upper:
                    type_path = 'Half'
                    cx = x
                    cy = y 
                    angle = angle

            angle = math.radians(-angle)
            print(area_ratio)
            font = cv.FONT_HERSHEY_SIMPLEX
            center = (int(x)-50, int(y))
            if not type_path == 'Not':
                cv.putText(res, type_path + '_' + str("%.2f" % cx) + '_' + str("%.2f" % cy) + '_' + str("%.2f" %
                                                                                                        angle) + '_' + str("%.2f" % area_ratio), center, font, 0.5, (0, 0, 255), 1, cv.LINE_AA)

        # # cv.imshow('img',img)
        publish_result(img, "bgr", "/path_img")
        publish_result(res, "bgr", "/path_result")

        # cv.imshow('res', res)
        # cv.imshow('dilate', dilate)

        # k = cv.waitKey(1) & 0xFF
        # if k == ord('q'):
        #     break

    # plt.show()

# def main():
#     global image, pressure

#     r = rospy.Rate(10)

#     while not rospy.is_shutdown():
#         # ret, frame = cap.read()
#         r.sleep()
#         if image is None:
#             print("image is none")
#             continue
        
#         bg_subtraction(image, pressure)

if __name__=='__main__':
    rospy.init_node('PathDetection')
    rospy.Subscriber("/vision/bottom/image_raw/compressed", CompressedImage, image_callback)
    # rospy.Subscriber("/bottom/left/image_raw/compressed", CompressedImage, image_callback)
    
    # rospy.Subscriber("/filter/pressure", HeaderFloat64, pressure_callback)
    # main()
    find_path()