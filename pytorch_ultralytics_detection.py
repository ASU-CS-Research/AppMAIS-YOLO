import torch
import cv2
import numpy as np


# im = torch.from_numpy(np.expand_dims(cv2.imread('images/image_AppMAIS3LB@2023-06-26@11-55-00.png').transpose(2, 0, 1), 0)).float()
im = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # load pretrained

results = model(im)  # inference
print(type(results))
# print(results.ims)  # array of original images (as np array) passed to model for inference
results.render()  # updates results.ims with boxes and labels
results.save()  # saves results.ims to results folder
