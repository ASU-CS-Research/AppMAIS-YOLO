from typing import Optional

import torch
import cv2
import os
import numpy as np
import ultralytics
from pytorch_yolov8 import PytorchYOLOV8


def get_frame_from_video(path: str, frame_ind: int):
    cap = cv2.VideoCapture(path)
    frame = None
    for i in range(frame_ind):
        ret, frame = cap.read()
    return frame


def load_model_pytorch(model_path=None, num_classes=None):
    if model_path is None:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, classes=num_classes, force_reload=True)
        print(model)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', path=model_path)
    # print(model)
    # model.model = model.model[:-1]
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


def run_predictions_on_video_ultralytics(model, video_filepath, show: Optional[bool] = False,
                                         output_filepath: Optional[str] = None):
    capture = cv2.VideoCapture(video_filepath)
    if not show:
        if output_filepath is None:
            output_filepath = f'./output_videos/'
            os.makedirs(output_filepath, exist_ok=True)
            output_filepath = os.path.join(output_filepath, f'ultralytics_{os.path.basename(video_filepath)[:-5]}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filepath, fourcc, 30, (640, 480))
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
    if not show:
        video_writer.release()


def run_predictions_on_video_pytorch(model, video_filepath: str, show: Optional[bool] = False,
                                     output_filepath: Optional[str] = None):
    model.eval()
    capture = cv2.VideoCapture(video_filepath)
    if not show:
        if output_filepath is None:
            output_filepath = f'./output_videos/'
            os.makedirs(output_filepath, exist_ok=True)
            output_filepath = os.path.join(output_filepath, f'pytorch_{os.path.basename(video_filepath)[:-5]}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filepath, fourcc, 30, (640, 480))
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        original_frame = frame.copy()
        frame = frame / 255.0
        frame = frame.transpose(2, 0, 1)
        frame = frame[None, :, :, :]
        frame_tensor = torch.from_numpy(frame)  # .unsqueeze(0)
        # print(frame_tensor.shape)
        frame_tensor = frame_tensor.type(torch.float32)
        results = model.predict(frame_tensor)
        # print(results)
        for predictions in results:
            # print(prediction.data.tolist())
            prediction_list = predictions.data.tolist()
            if not len(prediction_list) == 0:
                for prediction in prediction_list:
                    x1, y1, x2, y2, conf, class_id = prediction
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    color = (140, 230, 240) if class_id == 1 else (0, 0, 255)
                    cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 2)
        if show:
            cv2.imshow('frame', original_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            video_writer.write(original_frame)
    capture.release()
    if not show:
        video_writer.release()


if __name__ == '__main__':
    # model_path = os.path.abspath('./runs/detect/train7/weights/best.pt')
    video_filepath = os.path.abspath('videos/AppMAIS3LB@2023-06-26@11-55-00.h264')
    image_filepath = os.path.abspath('images/image_AppMAIS3LB@2023-06-26@11-55-00.png')
    # frame_ind = 120
    model = load_model_ultralytics(os.path.abspath('runs/detect/train7/weights/best.pt'))
    # print(type(model))
    # print(type(model.model))
    pytorch_sequential_model = model.model.__dict__["_modules"]["model"]
    pytorch_yaml_config = model.model.__dict__["yaml"]
    classes = model.model.__dict__["names"]
    print(classes)
    pytorch_yolo_from_yaml = PytorchYOLOV8(pytorch_sequential_model, pytorch_yaml_config, classes=None, conf_thresh=0.6,
                                           iou_thresh=0.6, max_det=20)
    # print(pytorch_yolo_from_yaml)
    # print(type(pytorch_sequential_model))
    # print(pytorch_sequential_model)
    # exit()
    run_predictions_on_video_pytorch(pytorch_yolo_from_yaml, video_filepath, show=False)
    run_predictions_on_video_ultralytics(model, video_filepath, show=False)
    # predictions = model.predict(image_filepath)
    # print(predictions)
    # num_boxes = len(predictions[0].boxes)
    # print(f'num_boxes: {num_boxes}')
    # model = load_model_pytorch(num_classes=2)
    # image = cv2.imread(image_filepath)
    # image = image / 255.0
    # image = image.transpose(2, 0, 1)
    # image_tensor = torch.from_numpy(image).unsqueeze(0)
    # image_tensor = image_tensor.type(torch.float32)
    # results = pytorch_yolo_from_yaml(image_tensor)
    # results = model(image_tensor)
    # print(results.shape)
    # print(f'results is a tuple, where results[0] is a torch of size: {results[0].shape}')
    # print(f'results[1] is a list of length {len(results[1])}')
    # print(f'results[1][0] is a torch of size: {results[1][0].shape}')
    # print(f'results[1][1] is a torch of size: {results[1][1].shape}')
    # print(f'results[1][2] is a torch of size: {results[1][2].shape}')
    # print(results[1])
    # print(len(results))
    # print(len(results[1]))
    # print(results[1][2].shape)

    # print(model)
    # run_predictions_on_video(model, video_filepath, show=False)
