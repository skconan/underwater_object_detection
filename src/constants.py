#!/usr/bin/env python
"""
    File name: constants.py
    Author: skconan
    Date created: 2019/04/09
    Python Version: 3.6
"""

import os
import platform

PATH_WS = os.path.dirname(os.path.abspath(__file__))
PATH_WS = os.path.dirname(PATH_WS)

if platform.system() == "Windows":
    PATH_DATASET = PATH_WS + r"\dataset"
    PATH_POOL = PATH_DATASET + r"\pool"
    PATH_ROBOSUB = PATH_DATASET + r"\robosub"
    PATH_OUTPUT = PATH_DATASET + r"\output"    
else:
    PATH_DATASET = PATH_WS + "/dataset"
    PATH_POOL = PATH_DATASET + "/pool"
    PATH_ROBOSUB = PATH_DATASET + "/robosub"
    PATH_OUTPUT = PATH_DATASET + "/output"    


COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_PINK = (255, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (71, 99, 255)
COLOR_VIOLET = (211, 0, 148)
IMAGE_FRONT_WIDTH = 1936
IMAGE_FRONT_HEIGHT = 1216
IMAGE_BOTTOM_WIDTH = 1800
IMAGE_BOTTOM_HEIGHT = 1100

MISSION_LIST = ['gate', 'buoy']