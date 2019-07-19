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


def load(img_file, label_file):
    img_in = tf.io.read_file(img_file)
    try:
        img = tf.image.decode_jpeg(img_in)
    except:
        img = tf.image.decode_png(img_in, channels=3)
    

    in_img = img

    img_in = tf.io.read_file(label_file)
    try:
        img = tf.image.decode_png(img_in, channels=3)
    except:
        img = tf.image.decode_jpeg(img_in)

    out_img = img

    in_img = tf.cast(in_img, tf.float32)
    out_img = tf.cast(out_img, tf.float32)

    return in_img, out_img


def resize(img, height, width):
    result = tf.image.resize(img, [height, width],
                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return result


def random_crop(in_img, out_img):
    if (out_img.shape[2] == 1):
        out_img = tf.image.grayscale_to_rgb(out_img)
    # add
    if (in_img.shape[2] == 1):
        in_img = tf.image.grayscale_to_rgb(in_img)

    stacked_image = tf.stack([in_img, out_img], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, 256, 256, 3])
    return cropped_image[0], cropped_image[1]


def normalize(img):
    # normalizing the images to [-1, 1]
    result = (img / 127.5) - 1
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


def random_jitter(in_img, out_img):
    # resizing to 286 x 286 x 3
    in_img = resize(in_img, 286, 286)
    out_img = resize(out_img, 286, 286)

    # randomly cropping to 256 x 256 x 3
    in_img, out_img = random_crop(in_img, out_img)

    val = tf.random.uniform(())
    if val > 0.5:
        # random mirroring
        in_img = tf.image.flip_left_right(in_img)
        out_img = tf.image.flip_left_right(out_img)

    return in_img, out_img

def apply_clahe(img_bgr):
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv.merge((l, a, b))
    res = cv.cvtColor(lab, cv.COLOR_Lab2BGR)
    return res

def load_image_train(img_file, label_file):
    in_img, out_img = load(img_file, label_file)
    in_img, out_img = random_jitter(in_img, out_img)

    in_img = normalize(in_img)
    out_img = normalize(out_img)
    return in_img, out_img


def load_image_test(img_file, label_file):
    in_img, out_img = load(img_file, label_file)

# add
    if (in_img.shape[2] == 1):
        in_img = tf.image.grayscale_to_rgb(in_img)

    in_img = resize(in_img, 256, 256)
    out_img = resize(out_img, 256, 256)

    in_img = normalize(in_img)
    out_img = normalize(out_img)

    return in_img, out_img


def load_image_predict(img_file):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_jpeg(img)
    in_img = img

    in_img = tf.cast(in_img, tf.float32)

    in_img = resize(in_img, 256, 256)
    in_img = normalize(in_img)

    return in_img


def rotation(img_tensor, degrees):
    img = np.uint8((img_tensor*0.5+0.5)*255)
    img = Image.fromarray(img)
    rotated = Image.Image.rotate(img, degrees)
    rotated = rotated.resize((280, 280))
    rotated = rotated.crop((12, 12, 268, 268))

    tensor = tf.convert_to_tensor(np.float32(rotated))
    tensor = normalize(tensor)
    return tensor
