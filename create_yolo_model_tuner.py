from ultralytics import YOLO
import os
data_yaml = os.path.abspath('./data.yaml')
model = YOLO('./yolov8s.pt')
model.tune(data=data_yaml, gpu_per_trial=1, train_args={ "epochs": 1})