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
import load_yolo_model as lym

video_filepath = os.path.abspath('videos/AppMAIS11R@2022-09-13@13-40-00.mp4')
model_path = os.path.abspath('/home/bee/bee-detection/trained_on_11s.pt')
frame_num = np.random.randint(120, 1700)
frame = lym.get_frame_from_video(video_filepath, frame_num)
model = lym.load_model_ultralytics(model_path)

low_conf_results = model(frame, conf=0.25)
just_right_conf_results = model(frame, conf=0.64)
high_conf_results = model(frame, conf=0.8)

def get_bounding_box_image(results, frame):
    # Get the bounding boxes from the results
    bounding_boxes = results[0].boxes
    boxed_frame = frame.copy()
    for box in bounding_boxes:
        x1, y1, x2, y2, conf, class_id = box.data.tolist()[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (140, 230, 240) if class_id == 1 else (0, 0, 255)
        cv2.rectangle(boxed_frame, (x1, y1), (x2, y2), color, 2)
        # Write the confidence on the image
        cv2.putText(boxed_frame, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return boxed_frame

low_boxed_frame = get_bounding_box_image(low_conf_results, frame)
just_right_boxed_frame = get_bounding_box_image(just_right_conf_results, frame)
high_boxed_frame = get_bounding_box_image(high_conf_results, frame)

# write the confidence threshold on the image
cv2.putText(low_boxed_frame, f'Confidence threshold: 0.25', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.putText(just_right_boxed_frame, f'Confidence threshold: 0.64', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.putText(high_boxed_frame, f'Confidence threshold: 0.8', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#save the 3 images in a folder called confidence_comparison
os.makedirs("confidence_comparison", exist_ok=True)
cv2.imwrite("confidence_comparison/low_conf.jpg", low_boxed_frame)
cv2.imwrite("confidence_comparison/just_right_conf.jpg", just_right_boxed_frame)
cv2.imwrite("confidence_comparison/high_conf.jpg", high_boxed_frame)
