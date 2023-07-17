import cv2
import pymongo
import os
import numpy as np
from loguru import logger

num_frames_to_retrieve = 200
frame_indices = np.random.rand(num_frames_to_retrieve) * 1794
output_path = os.path.abspath('./input_frames')

os.makedirs(output_path, exist_ok=True)

# Connect to MongoDB
client = pymongo.MongoClient()
db = client.beeDB
video_collection = db.VideoFiles
logger.debug('Connected to database...')

logger.info(f'Retrieving {num_frames_to_retrieve} videos from database')

# Query for number of video files equal to num_frames_to_retrieve at random
video_files_entries = video_collection.aggregate([{'$sample': {'size': num_frames_to_retrieve}}])

# Iterate through video files
for i, video_file_entry in enumerate(video_files_entries):
    frame_indices[i] = int(frame_indices[i])
    logger.debug(
        f'Processing video {i} from hive {video_file_entry["HiveName"]} at time {video_file_entry["TimeStamp"]}'
    )
    filepath = video_file_entry['FilePath']
    capture = cv2.VideoCapture(filepath)
    # Iterate through frames (setting frame does not work with .h264 files)
    frame = None
    logger.debug(f'Finding randomly selected frame index {frame_indices[i]} from video above.')
    for frame_ind in range(int(frame_indices[i])):
        ret, frame = capture.read()
        if frame is None:
            logger.warning(f'Frame {frame_indices[i]} not found in video {i}.')
            break
    if frame is not None:
        logger.info(f'Writing frame {frame_indices[i]} from video {i} to disk.')
        path = os.path.join(output_path, f'video_{os.path.basename(filepath[:-5])}_frame_{i}.png')
        cv2.imwrite(path, frame)


