
#!/usr/bin/env python
"""
    File name: bg_sub_by_kmean.py
    Author: skconan
    Date created: 2019/07/20
    Python Version: 2.7
    Reference: 
        https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html
        https://docs.opencv.org/master/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88
"""
import rospy
import cv2 as cv
import numpy as np
from utilities import *

max_iter = 5
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)

def normalize(gray):
    return np.uint8(255 * (gray - gray.min()) / (gray.max() - gray.min()))

def kmean(img, k):
    global criteria, max_iter
    Z = img.reshape((-1,1))

    # convert to np.float32
    Z = np.float32(Z)

    K = k
    ret,label,center=cv.kmeans(Z,K,None,criteria,max_iter,cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    result = res.reshape((img.shape))
    
    return result

def bg_subtraction(img, bg_k=1, fg_k=3, mode='neg'):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    gray,_,_ = cv.split(hsv)

    # start_time = rospy.Time.now()
    bg = kmean(gray, k=bg_k)
    fg = kmean(gray, k=fg_k)
 

    sub_sign = np.int16(fg) - np.int16(bg)
    
    if mode == 'neg':
        sub_neg = np.clip(sub_sign.copy(),sub_sign.copy().min(),0)
        sub_neg = normalize(sub_neg)
        _, result = cv.threshold(
            sub_neg, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )
    else:
        sub_pos = np.clip(sub_sign.copy(),0,sub_sign.copy().max())
        sub_pos = normalize(sub_pos)
        _, obj_pos = cv.threshold(
            result, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )

    # time_duration = rospy.Time.now()-start_time
    # print(time_duration.to_sec())

    return result

# if __name__ == "__main__":
#     rospy.init_node("test_bg_sub_kmean")
#     img = cv.imread("/media/skconan/SUPAKIT CPE/drawing_web_app/website/website/media/dataset/groundTruth/1543366588.png")
#     bg_subtraction(img)