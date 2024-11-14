import matplotlib.pyplot as plt
from scipy.stats import betabinom, binom, beta
import numpy as np
from typing import List, Optional, Tuple
import ultralytics
import os
import cv2 as cv
from betabinomial import beta_binom_on_data, parse_images_and_labels, rmse_on_data, get_caste_count_labels_results, \
    get_model_results

# data_path = os.path.abspath("/home/bee/bee-detection/data_appmais_lab/AppMAIS1s_labeled_data/val/")
# data_path = os.path.abspath('/home/bee/bee-detection/data_appmais_lab/AppMAIS11s_labeled_data/final_split_dataset/test')
# data_path = os.path.abspath('/home/bee/bee-detection/data_appmais_lab/AppMAIS1s_labeled_data/complete_data')
data_path = os.path.abspath('/home/bee/bee-detection/data_appmais_lab/stretch_test_2')
model_path = os.path.abspath("/home/bee/bee-detection/final_model.pt")
ultralytics_model = ultralytics.YOLO(model_path)
# ultralytics_model = ultralytics.YOLO("/home/bee/bee-detection/trained_on_11r_2022.pt")
# model_1s = ultralytics.YOLO("/home/bee/bee-detection/trained_on_11r_2022.pt")

images, images_filenames, labels_list = parse_images_and_labels(data_path)

output_directory = os.path.abspath('/home/bee/bee-detection/model_and_label_outputs/')

results, _, _ = get_model_results(ultralytics_model, images, labels_list, images_filenames)

print(results)
# exit(1)

# print(f"result 1 boxes: \n{results[0][0].boxes[0][0]}")
#
# print(f"label 1: \n{labels_list[0][0]}")


boxes = results[0].boxes
print(f"Box object:")
print(f"box: {boxes.cls}")
print(f"box: {boxes.xyxy}")

predictions = []

for img, cls, xyxy in zip(images, boxes.cls, boxes.xyxy):
    predictions.append({ "class": cls, "xyxy": xyxy})

print(predictions)

print(f"result[0].boxes: {results[0].boxes[0].data.tolist()[0]}")

