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
