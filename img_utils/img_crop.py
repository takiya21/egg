from pathlib import Path
# from yamaha_crop0 import yamaha_crop
import matplotlib.pyplot as plt

import os
import numpy as np
from PIL import Image
import cv2
import glob
import copy


def yamaha_crop(img_path, save_path):
    try:
        original = cv2.imread(img_path)
        h,w,c = original.shape
        original = original[int(h/2-1536):int(h/2+1536), int(w/2-1536):int(w/2+1536), :]#1536
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), original)
        print(f"{os.path.basename(path)} is ok")
    except RuntimeError as e:
        print(e)
        print("Load Error by {}".format(os.path.basename(img_path)))
        return np.zeros(0)


def yamaha_resize(img_path, save_path, width, height):
    try:
        original = cv2.imread(img_path)
        original = cv2.resize(original, (width, height))
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), original)
        print(f"{os.path.basename(path)} is ok")
    except RuntimeError as e:
        print(e)
        print("Load Error by {}".format(os.path.basename(img_path)))
        return np.zeros(0)


def yamaha_grayscale(img_path, save_path):
    try:
        original = cv2.imread(img_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), original)
        print(f"{os.path.basename(path)} is ok")
    except RuntimeError as e:
        print(e)
        print("Load Error by {}".format(os.path.basename(img_path)))
        return np.zeros(0)


if __name__ == '__main__':

    dir_paths = [

        #for crop
        
        "/dataset/dataset/egg/egg_preference50_resize320"
        #"/home/taki/egg/data/egg/test_input"
        


        #for resize
        # "/home/nakamura/yamaha/OK_A2_1117/OK_A2_1117_0001_0150_crop",

    ]

    save_paths = [

        #for crop
        
        "/home/taki/egg/data/egg/egg_preference_img_50/egg_preference50_resize256"
        


        #for resize256S


    ]

    for path in save_paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    for i in range(len(dir_paths)):
        print("\n" + dir_paths[i])
        paths = glob.glob(os.path.join(dir_paths[i], "*.jpg"))
        paths.sort()
        for path in paths:
            #yamaha_crop(path, save_paths[i])
            yamaha_resize(path, save_paths[i], width=256, height=256)
            # yamaha_grayscale(path, save_paths[i])