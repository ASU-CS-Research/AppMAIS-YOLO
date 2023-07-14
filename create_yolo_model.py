import ultralytics
import os

data_yaml = os.path.abspath("data.yaml")
pretrained_weights = os.path.abspath("yolov8s.pt")
model = ultralytics.YOLO(model='yolov8s.pt')
model.train(data='data.yaml', epochs=200, imgsz=(480, 640), verbose=True, batch=8)
model.val()
model.export()
