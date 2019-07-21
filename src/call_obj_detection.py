#!/usr/bin/env python
"""
    File name: call_obj_detection.py
    Author: skconan
    Date created: 2019/07/20
    Python Version: 2.7
"""

import rospy
import cv2 as cv
import numpy as np
from object_detection.srv import obj_detection_srv
from object_detection.msg import obj_detection_msg


rospy.init_node("call")
rospy.wait_for_service("object_detection_front")
call = rospy.ServiceProxy("object_detection_front", obj_detection_srv)
res = call(str("gate"))

arr = np.fromstring(res.data.mask.data, np.uint8)
image = cv.imdecode(arr, 1)
cv.imshow("img",image)
cv.waitKey(-1)