from typing import Optional

import torch
import cv2
import os
import numpy as np
import ultralytics


def get_frame_from_video(path: str, frame_ind: int):
    cap = cv2.VideoCapture(path)
    frame = None
    for i in range(frame_ind):
        ret, frame = cap.read()
    return frame


def load_model_pytorch(model_path=None, num_classes=None):
    if model_path is None:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, classes=num_classes)
        print(model)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', path=model_path)
    # print(model)
    model.eval()
    # model = model_and_info['model']
    # frame = get_frame_from_video(video_filepath, frame_ind)  # 120
    # frame = frame / 255.0
    # frame = frame.transpose(2, 0, 1)
    # frame_tensor = torch.from_numpy(frame).unsqueeze(0)
    # frame_tensor = frame_tensor.type(torch.float16)
    # results = model(frame_tensor)
    # results.print()
    return model


def load_model_ultralytics(model_path):
    model = ultralytics.YOLO(model_path)
    return model


def run_predictions_on_video_ultralytics(model, video_filepath, show: Optional[bool] = False):
    capture = cv2.VideoCapture(video_filepath)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('output.mp4', fourcc, 30, (480, 640))
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        predictions = model.predict(frame)
        results = predictions[0]
        bounding_boxes = results.boxes
        for box in bounding_boxes:
            x1, y1, x2, y2, conf, class_id = box.data.tolist()[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = (140, 230, 240) if class_id == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if show:
            cv2.imshow('frame', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            video_writer.write(frame)
    capture.release()
    video_writer.release()


if __name__ == '__main__':
    # model_path = os.path.abspath('./runs/detect/train7/weights/best.pt')
    video_filepath = os.path.abspath('videos/AppMAIS3LB@2023-06-26@11-55-00.h264')
    image_filepath = os.path.abspath('images/image_AppMAIS3LB@2023-06-26@11-55-00.png')
    # frame_ind = 120
    model = load_model_pytorch(num_classes=2)
    image = cv2.imread(image_filepath)
    image = image / 255.0
    image = image.transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    image_tensor = image_tensor.type(torch.float32)
    results = model(image_tensor)
    print(results[1])
    print(len(results))
    print(len(results[1]))
    print(results[1][2].shape)

    # print(model)
    # run_predictions_on_video(model, video_filepath, show=False)
