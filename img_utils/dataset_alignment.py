import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import os
import numpy as np
from PIL import Image
import cv2
import glob
import copy



def DetectionCircle(img_path, save_path):
    try:
        cimg = cv2.imread(img_path,0)

        #円を検出
        circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,100,param1=50,param2=30,minRadius=0,maxRadius=0)
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
            cv2.putText(cimg,"center:" + str(int(i[0]))+ ","+ str(int(i[1])),(i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
            cv2.putText(cimg,"r:" + str(int(i[2])),(i[0],i[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
            cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), cimg)
            
        #cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), cimg)
        print(f"{os.path.basename(path)} is ok")
    except RuntimeError as e:
        print(e)
        print("Load Error by {}".format(os.path.basename(img_path)))
        return np.zeros(0)


if __name__ == '__main__':

    dir_paths = [

        #for crop
        
        "/home/taki/egg/data/egg/test_input"
        #"/home/taki/egg/data/egg/test_input"
        


        #for resize
        # "/home/nakamura/yamaha/OK_A2_1117/OK_A2_1117_0001_0150_crop",

    ]

    save_paths = [

        #for crop
        
        "/home/taki/egg/data/egg/test_output"
        


        #for resize256S


    ]

    for path in save_paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    for i in range(len(dir_paths)):
        print("\n" + dir_paths[i])
        paths = glob.glob(os.path.join(dir_paths[i], "*.JPG"))
        paths.sort()

        for path in paths:
           DetectionCircle(path, save_paths[i])