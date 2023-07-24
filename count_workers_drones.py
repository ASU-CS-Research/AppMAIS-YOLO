import os
from typing import Optional
import cv2
import numpy as np
from tqdm import tqdm

from load_yolo_model import load_model_ultralytics


def count_classes_from_labels(path = "/home/bee/bee-detection/data_appmais_lab/AppMAIS11s_labeled_data/split_dataset/train/labels/"):
    labels = os.listdir(path)

    drones = 0
    workers = 0
    drone_files = []

    for label in labels:
        with open(f"{path}{label}", "r") as f:
            lines = f.readlines()
            for line in lines:
               class_index = line[0]
               if class_index == "0":
                    drones += 1
                    drone_files.append(label)
               elif class_index == "1":
                    workers += 1

    print("drones: ", drones)
    print("workers: ", workers)
    print("drone files: ", drone_files)




def count_classes_from_predictions(model, video_filepath, destination_video_path, show: Optional[bool] = False, max_frames: Optional[int] = None):
    capture = cv2.VideoCapture(video_filepath)
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    count = 0
    frames = []
    while True:
        ret, frame = capture.read()
        drone_count = 0
        worker_count = 0
        drone_counts = []
        worker_counts = []
        if frame is None:
            break
        predictions = model.predict(frame)
        results = predictions[0]
        bounding_boxes = results.boxes
        for box in bounding_boxes:
            x1, y1, x2, y2, conf, class_id = box.data.tolist()[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if class_id == 0:
                drone_count += 1
            elif class_id == 1:
                worker_count += 1
            color = (140, 230, 240) if class_id == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # write the counts on the frame in the upper left corner
        cv2.putText(frame, f"Predicted drone count: {drone_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f"Predicted worker count: {worker_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        drone_counts.append(drone_count)
        worker_counts.append(worker_count)

        drone_worker_ratio = np.average(np.array(drone_counts)) / (np.average(np.array(worker_counts)) + np.average(np.array(drone_counts)))

        cv2.putText(frame, f"Predicted drone/worker ratio: {drone_worker_ratio}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        frames.append(frame)
        count += 1
        if max_frames is not None and count >= max_frames:
            break
        if show:
            cv2.imshow('frame', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    capture.release()
    print(f'writing video with {len(frames)} frames...')
    video_writer = cv2.VideoWriter(filename = destination_video_path, fourcc = cv2.VideoWriter.fourcc(*'mp4v'), fps = 30, frameSize = (640, 480))
    for frame in tqdm(frames):
        video_writer.write(frame)
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model_path = os.path.abspath('./runs/detect/train7/weights/best11s.pt')
    video_filepath = os.path.abspath('videos/AppMAIS11R@2023-06-26@11-55-00.h264')
    frame_ind = 120
    model = load_model_ultralytics(model_path)
    count_classes_from_predictions(model, video_filepath, "output14.mp4", True, 1000)


