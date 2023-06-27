import torch
import cv2
import os
import numpy as np
import ultralytics


def load_yolo_model(model_path):
    model = torch.load(model_path)
    return model


def get_frame_from_video(path: str, frame_ind: int):
    cap = cv2.VideoCapture(path)
    frame = None
    for i in range(frame_ind):
        ret, frame = cap.read()
    return frame


def load_yolo_model_pytorch(model_path, video_filepath, frame_ind):
    model_and_info = load_yolo_model(model_path)
    model = model_and_info['model']
    model.eval()
    frame = get_frame_from_video(video_filepath, frame_ind)  # 120
    frame = frame / 255.0
    frame = frame.transpose(2, 0, 1)
    frame_tensor = torch.from_numpy(frame).unsqueeze(0)
    frame_tensor = frame_tensor.type(torch.float16)
    results = model(frame_tensor)
    results.print()


def load_model_ultralytics(model_path):
    model = ultralytics.YOLO(model_path)
    return model


def run_predictions_on_video(model, video_filepath):
    capture = cv2.VideoCapture(video_filepath)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        predictions = model.predict(frame)
        results = predictions[0]
        bounding_boxes = results.boxes
        for box in bounding_boxes:
            x1, y1, x2, y2, conf, class_id = box.data.tolist()[0]
            color = (140, 230, 240) if class_id == 0 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


if __name__ == '__main__':
    model_path = os.path.abspath('./runs/detect/train7/weights/best.pt')
    video_filepath = os.path.abspath('videos/AppMAIS3LB@2023-06-26@11-55-00.h264')
    frame_ind = 120
    load_model_ultralytics(model_path, video_filepath, frame_ind)
