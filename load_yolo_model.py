from typing import Optional
import PIL
import torch
import cv2
import os
import numpy as np
import ultralytics
import ffmpeg
from loguru import logger
import os

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


def run_predictions_on_video(model, video_filepath, show: Optional[bool] = False):
    capture = cv2.VideoCapture(video_filepath)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    count = 0
    frames = []
    while True:
        ret, frame = capture.read()
        if frame is None:
            #logger.warning("Frame is not currently available.")
            break
        predictions = model.predict(frame)
        results = predictions[0]
        bounding_boxes = results.boxes
        for box in bounding_boxes:
            x1, y1, x2, y2, conf, class_id = box.data.tolist()[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = (140, 230, 240) if class_id == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        #frame_filename = f"videoframes/frame%{count}d.jpg"
        #cv2.imwrite(frame_filename, frame)
        #count += 1

        if count == 60:
            break

        count += 1

        frames.append(frame)

        if show:
            cv2.imshow('frame', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        #else:
            #video_writer.write(frame)

    video_writer = cv2.VideoWriter('output7.mp4', fourcc, 30, (480, 640))
    for frame in frames:
        video_writer.write(frame)

    capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = os.path.abspath('./runs/detect/train7/weights/best.pt')
    video_filepath = os.path.abspath('videos/AppMAIS3LB@2023-06-26@11-55-00.h264')
    frame_ind = 120
    model = load_model_ultralytics(model_path)
    run_predictions_on_video(model, video_filepath, show=False)
    #os.system("ffmpeg -f image2 -r 1/0.3 -i ./videoframes/* -vcodec mpeg4 -y ./videos/3LB_2023-06-26@11-55-00_output.mp4")

