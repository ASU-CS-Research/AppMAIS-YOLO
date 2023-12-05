from typing import Optional
import PIL
import torch
import cv2
import os
import numpy as np
import ultralytics
import os
#from skvideo.io import FFmpegWriter
from tqdm import tqdm
from sklearn.utils import gen_batches


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


def draw_boxes_on_frame(frame, results):
    bounding_boxes = results.boxes
    for box in bounding_boxes:
        x1, y1, x2, y2, conf, class_id = box.data.tolist()[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        colors = ((0, 180, 255), (255, 180, 0))
        color = colors[int(class_id)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame


def run_predictions_on_video(model, video_filepath, destination_video_path, show: Optional[bool] = False,
                             max_frames: Optional[int] = None, batch_size: Optional[int] = 16,
                             conf: Optional[float] = 0.2):
    capture = cv2.VideoCapture(video_filepath)
    count = 0
    frames = []
    edited_frames = []
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        frames.append(frame)
    print(f'Total frames: {len(frames)}')
    frames = np.array(frames)
    batches = gen_batches(len(frames), batch_size)
    for batch in tqdm(batches):
        predictions = model.predict(list(frames[batch]), conf=conf)
        for frame, results in zip(frames[batch], predictions):
            frame = draw_boxes_on_frame(frame, results)
            edited_frames.append(frame)
            count += 1
            if max_frames is not None and count >= max_frames:
                break
            if show:
                cv2.imshow('image', frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

    # predictions = model.predict(frames)
    #
    # for frame, results in zip(frames, predictions):
    #     bounding_boxes = results.boxes
    #     for box in bounding_boxes:
    #         x1, y1, x2, y2, conf, class_id = box.data.tolist()[0]
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #         color = (140, 230, 240) if class_id == 1 else (0, 0, 255)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    #
    #     edited_frames.append(frame)
    #     count += 1
    #     if max_frames is not None and count >= max_frames:
    #         break
    #     if show:
    #         cv2.imshow('image', frame)
    #         if cv2.waitKey(30) & 0xFF == ord('q'):
    #             break
    capture.release()
    print(f'writing video with {len(frames)} frames...')
    video_writer = cv2.VideoWriter(
        filename=destination_video_path, fourcc=cv2.VideoWriter.fourcc(*'mp4v'), fps=30, frameSize=(640, 480)
    )
    for frame in tqdm(edited_frames):
        video_writer.write(frame)
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model_path = os.path.abspath('/home/bee/bee-detection/final_model.pt')
    video_filepath = os.path.abspath('videos/AppMAIS11R@2022-09-13@13-40-00.h264')
    frame_ind = 248
    model = load_model_ultralytics(model_path)
    # destination_video_path = os.path.abspath('output21.mp4')
    # run_predictions_on_video(model = model, video_filepath = video_filepath, destination_video_path = destination_video_path, show=False)
    desired_frame = get_frame_from_video(video_filepath, frame_ind)
    desired_frame_2 = desired_frame.copy()
    desired_frame_64 = desired_frame.copy()
    desired_frame_8 = desired_frame.copy()
    predictions2 = model.predict(desired_frame_2, conf=0.2)
    predictions64 = model.predict(desired_frame_64, conf=0.64)
    predictions8 = model.predict(desired_frame_8, conf=0.8)
    conf2frame = draw_boxes_on_frame(desired_frame_2, predictions2[0])
    conf64frame = draw_boxes_on_frame(desired_frame_64, predictions64[0])
    conf8frame = draw_boxes_on_frame(desired_frame_8, predictions8[0])
    # Save the frames
    cv2.imwrite('conf2frame.png', conf2frame)
    cv2.imwrite('conf64frame.png', conf64frame)
    cv2.imwrite('conf8frame.png', conf8frame)

