import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("/home/olofintuyita/Desktop/yolo/labeling/3rd_set_both11s/images/video_AppMAIS11R@2022-05-15@15-00-00_frame_126.png")

f = open("/home/olofintuyita/Desktop/yolo/labeling/3rd_set_both11s/labels_list/video_AppMAIS11R@2022-05-15@15-00-00_frame_126.txt", "r").readlines()

for line in f:
    if line[0] == "0":
        x = float(line[2:9])
        y = float(line[11:18])
        w = float(line[20:27])
        h = float(line[29:36])
        print(img[int(x-w/2):int(x+w/2), int(y-h/2) : int(y+h/2)])
        section = img[int(x-w/2):int(x+w/2), int(y-h/2) : int(y+h/2)]
        print(section)

