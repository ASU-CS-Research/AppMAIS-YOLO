import cv2
import pymongo
import os
import numpy as np
from loguru import logger
from datetime import datetime

num_frames_to_retrieve = 100
frame_indices = np.random.rand(num_frames_to_retrieve) * 1794
output_path = os.path.abspath("./even_more_11s_frames")

os.makedirs(output_path, exist_ok=True)

# Connect to MongoDB
client = pymongo.MongoClient()
db = client.beeDB
video_collection = db.VideoFiles
logger.debug('Connected to database...')

logger.info(f'Retrieving {num_frames_to_retrieve} videos from database')

# Query for number of video files equal to num_frames_to_retrieve at random
# video_files_entries = video_collection.aggregate([{'$sample': {'size': num_frames_to_retrieve}}])
# Here's an example of a hard-coded query for a specific time range for a few specific hives
date_string_format = "%Y-%m-%d"
# aggregation_pipeline = [ {'$match': {'HiveName': {'$in': ['AppMAIS11R']}}},
#     {'$match':
#         {'TimeStamp': {'$gt': datetime.strptime("2022-09-01", date_string_format),
#                        '$lt': datetime.strptime("2022-11-30", date_string_format)}}
#         },
# {'$match': {
#      '$expr': {
#         '$and': [
#             {'$gte': [{'$hour': "$TimeStamp"}, 12]},
#             {'$lt':  [{'$hour': "$TimeStamp"}, 16]}
#         ]
#     }}},
#                          # {'$match': {"Tag": {"$eq": "Drone"}}},
#     {'$sample': {'size': num_frames_to_retrieve}}
# ]
aggregation_pipeline = [
    {"$match": {"HiveName": {'$in': ['AppMAIS11R', 'AppMAIS11L', 'AppMAIS11RB', 'AppMAIS11LB']}}},
    {'$match': {
         '$expr': {
            '$and': [
                {'$gte': [{'$hour': "$TimeStamp"}, 12]},
                {'$lt':  [{'$hour': "$TimeStamp"}, 16]}
            ]
        }}},
    {"$sample": {'size': num_frames_to_retrieve}}
]
video_files_entries = video_collection.aggregate(aggregation_pipeline)

logger.info(f'Cursor is alive: {video_files_entries.alive}')

# Iterate through video files
for i, video_file_entry in enumerate(video_files_entries):
    frame_indices[i] = int(frame_indices[i])
    logger.debug(
        f'Processing video {i} from hive {video_file_entry["HiveName"]}'
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
        path = os.path.join(output_path, f'video_{os.path.basename(filepath[:-5])}_frame_{int(frame_indices[i])}.png')
        cv2.imwrite(path, frame)
    else:
        logger.warning(f'Frame {frame_indices[i]} not found in video {i}.')


