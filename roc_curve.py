from pprint import pprint

import matplotlib.pyplot as plt
from reportlab.graphics.charts.barcharts import sampleSymbol1
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
# data_path = os.path.abspath('/home/bee/bee-detection/data_appmais_lab/stretch_test')
# data_path = os.path.abspath('/home/bee/bee-detection/data_appmais_lab/AppMAIS11s_labeled_data/final_split_dataset/test')
# data_path = os.path.abspath('/home/bee/bee-detection/data_appmais_lab/AppMAIS11s_labeled_data')
# data_path = os.path.abspath('/home/bee/bee-detection/data_appmais_lab/AppMAIS11s_labeled_data/')

model_path = os.path.abspath("/home/bee/bee-detection/final_model.pt")
ultralytics_model = ultralytics.YOLO(model_path)
# ultralytics_model = ultralytics.YOLO("/home/bee/bee-detection/trained_on_11r_2022.pt")
# model_1s = ultralytics.YOLO("/home/bee/bee-detection/trained_on_11r_2022.pt")

images, images_filenames, labels_list = parse_images_and_labels(data_path)

# output_directory = os.path.abspath('/home/bee/bee-detection/model_and_label_outputs/')

# results, _, _ = get_model_results(ultralytics_model, images, labels_list, images_filenames)

# print(results)
# exit(1)

# print(f"result 1 boxes: \n{results[0][0].boxes[0][0]}")
#
# print(f"label 1: \n{labels_list[0][0]}")


# boxes = results[0].boxes
# print(f"Box object:")
# print(f"box: {boxes.cls}")
# print(f"box: {boxes.xyxy}")
#
# exit(0)

# predictions = []
#
# for img, cls, xyxy in zip(images, boxes.cls, boxes.xyxy):
#     predictions.append({ "class": cls, "xyxy": xyxy})

# print(predictions)

# print(f"result[0].boxes: {results[0].boxes[0].data.tolist()[0]}")

labels_xyxy = []
for label_img in labels_list:
    label_xyxy = []
    for label in labels_list[0]:
        class_id, center_x, center_y, w, h = label
        x1 = int((center_x - w / 2) * images[0].shape[1])
        y1 = int((center_y - h / 2) * images[0].shape[0])
        x2 = int((center_x + w / 2) * images[0].shape[1])
        y2 = int((center_y + h / 2) * images[0].shape[0])
        label_xyxy.append([class_id, x1, y1, x2, y2])

# print(f"\nlabels_list[0]: \n{labels_list[0][0]}")
# print(f"\nlabels_xyxy: \n{labels_xyxy[0][0]}")
# print(f"box 1: {predictions[0]}")

def iou(box1: List[int], box2: List[int]) -> float:
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x5, y5 = max(x1, x3), max(y1, y3)
    x6, y6 = min(x2, x4), min(y2, y4)
    intersection = max(0, x6 - x5) * max(0, y6 - y5)
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
    return intersection / union

label_prediction_match_by_image = []

# for label_xyxy, prediction in zip(labels_xyxy, predictions):
#     label_prediction_match = {}
#
#     for label in labels_xyxy:
#         label_prediction_match[label] = None
#
#     for prediction in predictions:
#         iou_scores = [iou(label[1:], prediction["xyxy"]) for label in labels_xyxy]
#
#         idx_max_iou = np.argmax(iou_scores)
#         if iou_scores[idx_max_iou] > 0.5:
#             label_prediction_match[labels_xyxy[idx_max_iou]] = prediction


def get_label_pred_counts_for_conf(conf: float) -> List[List[int]]:
    results = ultralytics_model.predict(images, conf=conf)
    label_pred_counts = []
    for sample in zip(labels_list,results):
        labels, result = sample
        (worker_true_count, drone_true_count), (worker_pred_count, drone_pred_count) = get_caste_count_labels_results(labels, result)
        label_pred_counts.append([worker_true_count, drone_true_count, worker_pred_count, drone_pred_count])
    return label_pred_counts

# label_pred_counts = []
# for sample in zip(labels_list,results):
#     labels, result = sample
#     (worker_true_count, drone_true_count), (worker_pred_count, drone_pred_count) = get_caste_count_labels_results(labels, result)
#     label_pred_counts.append([worker_true_count, drone_true_count, worker_pred_count, drone_pred_count])

# def get_drone_tpr_fpr(label_pred_counts: List[List[int]]) -> Tuple[List[float], List[float]]:
#     tp = 0
#     fp = 0
#     samples = 0
#     for label_pred_count in label_pred_counts:
#         worker_true_count, drone_true_count, worker_pred_count, drone_pred_count = label_pred_count
#         tp += min (drone_true_count, drone_pred_count)
#         fp += max(0, drone_pred_count - drone_true_count)
#         samples += max(drone_true_count, drone_pred_count)
#     tpr = tp / samples
#     fpr = fp / samples
#     return tpr, fpr
#
# def get_worker_tpr_fpr(label_pred_counts: List[List[int]]) -> Tuple[List[float], List[float]]:
#     tp = 0
#     fp = 0
#     samples = 0
#     for label_pred_count in label_pred_counts:
#         worker_true_count, drone_true_count, worker_pred_count, drone_pred_count = label_pred_count
#         tp += min (worker_true_count, worker_pred_count)
#         fp += max(0, worker_pred_count - worker_true_count)
#         samples += max(worker_true_count, worker_pred_count)
#     tpr = tp / samples
#     fpr = fp / samples
#     return tpr, fpr

def get_tpr_fpr(label_pred_counts: List[List[int]]):
    drone_tp = 0
    drone_fp = 0
    worker_tp = 0
    worker_fp = 0
    drone_samples = 0
    worker_samples = 0
    for label_pred_count in label_pred_counts:
        worker_true_count, drone_true_count, worker_pred_count, drone_pred_count = label_pred_count
        drone_tp += min (drone_true_count, drone_pred_count)
        drone_fp += max(0, drone_pred_count - drone_true_count)
        drone_samples += max(drone_true_count, drone_pred_count)
        worker_tp += min (worker_true_count, worker_pred_count)
        worker_fp += max(0, worker_pred_count - worker_true_count)
        worker_samples += max(worker_true_count, worker_pred_count)
    drone_tpr = drone_tp / drone_samples
    drone_fpr = drone_fp / drone_samples
    worker_tpr = worker_tp / worker_samples
    worker_fpr = worker_fp / worker_samples
    return drone_tpr, drone_fpr, worker_tpr, worker_fpr

def get_tpr_fpr_by_iou_by_image(image_labels, image_results):
    boxes_xyxy = [image_result.boxes[0][0].data.tolist()[0] for image_result in image_results]
    boxes_cls = [image_result.boxes[0][1].data.tolist()[0] for image_result in image_results]
    boxes = [[cls, *xyxy] for cls, xyxy in zip(boxes_cls, boxes_xyxy)]

    # boxes = list of elements like [cls, x1, y1, x2, y2], predicted by the model
    # image_labels = list of elements like [cls, x1, y1, x2, y2], ground truth

    tp = 0
    fp = 0
    samples = 0
    predicted_labels = [0 for _ in range(len(image_labels))]

    for box in boxes:
        samples += 1
        iou_scores = [iou(box[1:], label[1:]) for label in image_labels]
        idx_max_iou = np.argmax(iou_scores)

        class OutOfLabels(Exception):
            pass

        try:
            while predicted_labels[idx_max_iou] == 1:
                iou_scores[idx_max_iou] = 0
                try:
                    idx_max_iou = np.argmax(iou_scores)
                except ValueError:
                    # no more labels to match
                    # the model predicted more boxes than there are labels
                    fp += 1
                    raise OutOfLabels
        except OutOfLabels:
            continue

        if iou_scores[idx_max_iou] < 0.5 or box[0] != image_labels[idx_max_iou][0]:
            fp += 1
        else:
            predicted_labels[idx_max_iou] = 1
            tp += 1




        # if iou_scores[idx_max_iou] > 0.5:
        #     if predicted_labels[idx_max_iou] == 0:
        #     if box[0] == 0:
        #         tp += 1
        #     else:
        #         fp += 1
        # else:
        #     if box[0] == 0:
        #         fp += 1

def area_under_curve(tpr: List[float], fpr: List[float]) -> float:
    auc = 0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
    return auc

conf_and_rates = []

confs = np.arange(0.0, 1.0, 0.05)
# reverse confs
confs = confs[::-1]
for conf in confs:
    label_pred_counts = get_label_pred_counts_for_conf(conf)
    drone_tpr, drone_fpr, worker_tpr, worker_fpr = get_tpr_fpr(label_pred_counts)
    print(f"conf: {conf} drone_tpr: {drone_tpr} drone_fpr: {drone_fpr} worker_tpr: {worker_tpr} worker_fpr: {worker_fpr}")
    conf_and_rates.append([conf, drone_tpr, drone_fpr, worker_tpr, worker_fpr])

pprint(conf_and_rates)

conf_and_rates = np.array(conf_and_rates)

plt.plot(conf_and_rates[:, 2], conf_and_rates[:, 1], label="drone")
# annotate the conf for each point
for conf, drone_tpr, drone_fpr, worker_tpr, worker_fpr in conf_and_rates:
    plt.annotate(f"{conf:.2f}", (drone_fpr, drone_tpr))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Drone ROC curve")
plt.show()

plt.plot(conf_and_rates[:, 4], conf_and_rates[:, 3], label="worker")
# annotate the conf for each point
for conf, drone_tpr, drone_fpr, worker_tpr, worker_fpr in conf_and_rates:
    plt.annotate(f"{conf:.2f}", (worker_fpr, worker_tpr))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Worker ROC curve")

auc_drone = area_under_curve(conf_and_rates[:, 1], conf_and_rates[:, 2])
auc_worker = area_under_curve(conf_and_rates[:, 3], conf_and_rates[:, 4])

plt.show()

print(f"Drone AUC: {auc_drone}")
print(f"Worker AUC: {auc_worker}")
#  why is the output of the area under curve function negative?
#  the output of the area under curve function is negative because the fpr and tpr values are not sorted in ascending order.
print(f"conf_and_rates is conf, drone_tpr, drone_fpr, worker_tpr, worker_fpr")
pprint(conf_and_rates)
