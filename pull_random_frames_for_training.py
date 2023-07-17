import cv2
import pymongo
import os
import numpy as np

num_frames_to_retrieve = 10
frame_indices = np.random.rand(num_frames_to_retrieve) * 1794
output_path = os.path.abspath('./input_frames')

# Connect to MongoDB
client = pymongo.MongoClient()
db = client.beeDB
video_collection = db.VideoFiles

# Query for number of video files equal to num_frames_to_retrieve at random
video_files_entries = video_collection.aggregate([{'$sample': {'size': num_frames_to_retrieve}}])

# Iterate through video files
for i, video_file_entry in enumerate(video_files_entries):
    filepath = video_file_entry['FilePath']
    capture = cv2.VideoCapture(filepath)
    # Iterate through frames (setting frame does not work with .h264 files)
    frame = None
    for frame_ind in range(int(frame_indices[i])):
        ret, frame = capture.read()
        if frame is None:
            break
    if frame is not None:
        cv2.imwrite(os.path.join(output_path, f'video_{filepath[:-5]}_frame_{i}.png'), frame)


