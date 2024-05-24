from typing import Optional, List, Tuple

import ultralytics
import os
from datetime import datetime, time
from MongoUtils.mongo_helper import MongoHelper
from pymongo import MongoClient
import json
from loguru import logger
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum


class DetectionClasses(Enum):
    """
    An enumeration of the detection classes for the YOLO model. This is used to determine the number of drones and
    workers detected in a frame.
    """
    DRONE = 0
    WORKER = 1


class YOLOModel:
    """
    A class to represent a YOLO model. This class is a wrapper around the ultralytics.YOLO class, with the purpose of
    connecting to the database and getting a representation of
    """
    def __init__(self, pretrained_weights_path: str, mongo_client: MongoClient,
                 confidence_threshold: Optional[float] = 0.64,
                 batch_size: Optional[int] = 64, video_length_frames: Optional[int] = 1794):
        logger.info(f"Initializing YOLO model with pretrained weights at {pretrained_weights_path}.")
        self._model = ultralytics.YOLO(model=pretrained_weights_path)
        self._mongo_client = mongo_client
        self._bee_db = self._mongo_client.beeDB
        self._video_files_collection = self._bee_db.VideoFiles
        self._hive_collection = self._bee_db.HiveWorkspace
        self._drone_worker_count_collection = self._bee_db.DroneWorkerCount
        self._confidence_threshold = confidence_threshold
        self._batch_size = batch_size
        self._video_length_frames = video_length_frames
        self._desired_frame_ind = self._video_length_frames // 2


    def run_model_on_videos(self, start_date: datetime, end_date: datetime, hive_list: Optional[List[str]] = None,
                            start_time: Optional[time] = None, end_time: Optional[time] = None,
                            upload_to_mongo: Optional[bool] = True, stride: Optional[int] = 1) -> (
            pd.DataFrame):
        """
        Retrieves random frames from all videos listed in the date range for the given hives, and runs the YOLO model
        on those frames. Number of drones and workers are returned.
        Args:
            start_date (datetime): The start date for the videos.
            end_date (datetime): The end date for the videos.
            hive_list (Optional[List[str]]): A list of hives to run the model on. If None, all active hives in the date
              range will be used.
            start_time (Optional[time]): The start time for the videos. If None, all times will be used. This is used
              to filter the videos, such as a start time of 12pm means that any day's videos before 12pm will be
                ignored.
            end_time (Optional[time]): The end time for the videos. If None, all times will be used. This is used
              to filter the videos, such as an end time of 4pm means that any day's videos after 4pm will be ignored.
            upload_to_mongo (Optional[bool]): Whether to upload the results to the database. Default is True.
            stride (Optional[int]): Every ``stride`` videos will be returned. Default is 1, which means that every video
              that matches the other criteria will be returned.

        Returns:
            pd.DataFrame: A DataFrame with the number of drones and workers detected in each frame. The columns are:
              - HiveName: The name of the hive.
              - PopulationMarker: The population marker for the hive.
              - FrameNumber: The frame number.
              - TimeStamp: The video time stamp.
              - NumDrones: The number of drones detected in the frame.
              - NumWorkers: The number of workers detected in the frame.
              - DroneToWorkerRatio: The ratio of drones to workers detected in the frame.
        """
        if hive_list is None:
            hive_list = self.get_active_hives_in_time_frame(start_date=start_date, end_date=end_date)
        logger.info(f"Running YOLO model on videos for hives {hive_list} between {start_date} and {end_date}.")
        results = []
        already_processed_frames = []
        for hive in hive_list:
            hive_name, population_marker = self.get_population_marker_from_hive_name(hive)
            video_filepaths = self.get_video_filepaths_for_hive(
                hivename=hive_name, start_date=start_date, end_date=end_date, start_time=start_time, end_time=end_time,
                stride=stride
            )
            already_processed_frames_for_hive, video_filepaths = self._find_already_processed_frames(
                video_filepaths=video_filepaths
            )
            already_processed_frames += already_processed_frames_for_hive
            frame_related_data = []
            frames = []
            for video_filepath in video_filepaths:
                date_str, time_str = video_filepath.split("@")[1], video_filepath.split("@")[2].split('.')[0]
                timestamp = datetime.strptime(f"{date_str}@{time_str}", "%Y-%m-%d@%H-%M-%S")
                frame = self._retrieve_middle_frame_from_video(video_filepath, self._desired_frame_ind)
                if frame is None:
                    continue
                frame_related_data.append({
                    "HiveName": hive_name,
                    "PopulationMarker": population_marker,
                    "FrameNumber": self._desired_frame_ind,
                    "TimeStamp": timestamp
                })
                frames.append(frame)
            frames_in_batches = [frames[i:i + self._batch_size] for i in range(0, len(frames), self._batch_size)]
            for i, batch in enumerate(frames_in_batches):
                predictions = self._model.predict(batch, conf=self._confidence_threshold, save=False, stream=True)
                for j, prediction in enumerate(predictions):
                    frame_data = frame_related_data[i * self._batch_size + j]  # Getting the frame data for the frame
                    num_drones = 0
                    num_workers = 0
                    for label in prediction.boxes.cls.cpu().numpy():
                        if label == DetectionClasses.DRONE.value:
                            num_drones += 1
                        elif label == DetectionClasses.WORKER.value:
                            num_workers += 1
                    results.append({
                        "HiveName": frame_data["HiveName"],
                        "PopulationMarker": frame_data["PopulationMarker"],
                        "FrameNumber": frame_data["FrameNumber"],
                        "TimeStamp": frame_data["TimeStamp"],
                        "NumDrones": num_drones,
                        "NumWorkers": num_workers,
                        "DroneToWorkerRatio": num_drones / num_workers if num_workers != 0 else np.nan,
                        "FilePath": video_filepaths[i * self._batch_size + j]
                    })
        if upload_to_mongo and len(results) > 0:
            logger.info("Uploading results to the database.")
            self._drone_worker_count_collection.insert_many(results)
        results += already_processed_frames
        return pd.DataFrame(results)


    @staticmethod
    def graph_model_results_for_hive(results: pd.DataFrame, hive_names: List[str],
                                     alpha: Optional[float]=0.7, figsize: Optional[Tuple[int, int]] = (20, 20),
                                     font_size: Optional[int] = 22, markersize: Optional[int] = 10) -> None:
        """
        Graphs the number of drones and workers detected in each frame for the given hive.

        Args:
            results (pd.DataFrame): The DataFrame containing the results of the model.
            hive_names (List[str]): The names of the hive to graph the results for.
            alpha (Optional[float]): The alpha value for the scatter plot. Default is 0.7.
        """
        # Make sure there is at least one entry in the DataFrame
        if results.shape[0] == 0:
            logger.warning("No results found in the DataFrame. Exiting.")
            return
        # Set the text size pretty large for readability
        plt.rcParams.update({'font.size': font_size})
        # increase point size a little as well
        plt.rcParams.update({'lines.markersize': markersize})
        fig, ax = plt.subplots(figsize=figsize)
        for hive_name in hive_names:
            hive_name, population_marker = YOLOModel.get_population_marker_from_hive_name(hive_name)
            hive_results = results[results["HiveName"] == hive_name]
            hive_results = hive_results[hive_results["PopulationMarker"] == population_marker]
            hive_label = f"{hive_name}{population_marker}" if population_marker != 'A' else hive_name
            ax.scatter(hive_results["TimeStamp"], hive_results["DroneToWorkerRatio"],
                       label=hive_label, alpha=alpha)
        ax.set_xlabel("Time Stamp")
        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45)
        ax.set_ylabel("Drone to Worker Ratio")
        ax.set_title(f"Number of Drones and Workers Detected")
        # Add a legend on the outside of the plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

    """
    The following methods may not be best for the model class, but leaving them here as they're currently needed.
    """

    @staticmethod
    def get_population_marker_from_hive_name(hive_name: str) -> Tuple[str, str]:
        if hive_name[-1] not in ['L', 'R']:
            hive_name_without_population_marker = hive_name[:-1]
            population_marker = hive_name[-1]
        else:
            hive_name_without_population_marker = hive_name
            population_marker = 'A'
        return hive_name_without_population_marker, population_marker

    def _find_already_processed_frames(self, video_filepaths):
        """
        Finds the frames that have already been processed for the given set of video filepaths. Uses
        ``self._desired_frame_ind`` to find the frame number to check for.
        Args:
            video_filepaths: The filepaths for the videos to check for processed frames.

        Returns:
            Tuple[List[dict], List[str]]: A list of dictionaries containing the processed frame data, and a list of
              filepaths from ``video_filepaths`` that were not found in the database.
        """
        query = {
            "FrameNumber": self._desired_frame_ind,
            "FilePath": { "$in": video_filepaths }
        }
        results = []
        processed_frames_cursor = self._drone_worker_count_collection.find(query)
        for document in processed_frames_cursor:
            video_filepaths.remove(document["FilePath"])
            result = {
                "HiveName": document["HiveName"],
                "PopulationMarker": document["PopulationMarker"],
                "FrameNumber": document["FrameNumber"],
                "TimeStamp": document["TimeStamp"],
                "NumDrones": document["NumDrones"],
                "NumWorkers": document["NumWorkers"],
                "DroneToWorkerRatio": document["DroneToWorkerRatio"],
                "FilePath": document["FilePath"]
            }
            results.append(result)
        return results, video_filepaths


    def get_video_filepaths_for_hive(self, hivename: str, num_videos: Optional[int] = None,
                                     start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                                     start_time: Optional[time] = None, end_time: Optional[time] = None,
                                     stride: Optional[int] = 1) -> List[str]:
        """
        Returns a list of filepaths for the videos associated with the given hive. These are converted using the /mnt/
        directory structure on lambda.
        Args:
            hivename (str): The name of the hive to get the videos for.
            num_videos (Optional[int]): The number of videos to retrieve (in order of recording time). If None, all
              videos will be used.
            start_date (Optional[datetime]): The start date for the videos. If None, the start date will be the
              earliest date that the hive was active.
            end_date (Optional[datetime]): The end date for the videos. If None, the end date will be the latest
              date that the hive was active.
            start_time (Optional[time]): The start time for the videos. If None, all times will be used. This is used
              to filter the videos, such as a start time of 12pm means that any day's videos before 12pm will be
              ignored.
            end_time (Optional[time]): The end time for the videos. If None, all times will be used. This is used
              to filter the videos, such as an end time of 4pm means that any day's videos after 4pm will be ignored.
            stride (Optional[int]): Every ``stride`` videos will be returned. Default is 1, which means that every video
              that matches the other criteria will be returned.

        Returns:
            List[str]: A list of filepaths for the videos associated with the given hive.
        """
        logger.debug(f"Querying for {'all' if num_videos is None else num_videos} video filepaths for hive "
                     f"{hivename} between {start_date} and {end_date}.")
        query = { "HiveName": hivename }
        if start_date is not None or end_date is not None:
            query["TimeStamp"] = {}
        if start_date is not None:
            query["TimeStamp"].update({"$gte": start_date})
        if end_date is not None:
            query["TimeStamp"].update({"$lte": end_date})
        videos_cursor = self._video_files_collection.find(query)
        filepaths = []
        i = -1
        for video_document in videos_cursor:
            if start_time is not None or end_time is not None:
                video_time = video_document["TimeStamp"].time()
                if start_time is not None and video_time < start_time:
                    continue
                if end_time is not None and video_time > end_time:
                    continue
            i += 1
            if i % stride != 0:
                continue
            filepath = video_document["FilePath"]
            filepath = filepath.replace("/usr/local/bee", "/mnt")
            if os.path.exists(filepath):
                filepaths.append(filepath)
            elif os.path.exists(filepath.replace("h264", "mp4")):  # Needs to be done in this transitional period before
                filepaths.append(filepath.replace("h264", "mp4"))  # all videos are stored as mp4
            else:
                logger.warning(f"Could not find video at {filepath}. Skipping.")
            if num_videos is not None and len(filepaths) == num_videos:
                break
        if num_videos is not None and len(filepaths) < num_videos:
            logger.warning(f"Could not find {num_videos} videos for hive {hivename}. Found {len(filepaths)} videos.")
        return filepaths

    def get_active_hives_in_time_frame(self, start_date: datetime, end_date: datetime) -> List[str]:
        """
        Returns a list of active hives in the given time frame.

        Args:
            start_date (Optional[datetime]): The start date for the time frame.
            end_date (Optional[datetime]): The end date for the time frame.

        Returns:
            List[str]: A list of active hives in the given time frame.
        """
        # Using two time frames (startA, endA) and (startB, endB), we can find whether they overlap by checking:
        # startA <= endB and startB <= endA
        active_hives = []
        hives_cursor = self._hive_collection.find({})
        for hive in hives_cursor:
            hive_name = hive["HiveName"]
            bee_install_date = hive["BeeInstallDate"]
            if bee_install_date <= end_date:
                hive_name_with_population_marker = (f"{hive_name}{hive['PopulationMarker']}"
                                                    if hive["PopulationMarker"] != 'A' else hive_name)
                active_hives.append(hive_name_with_population_marker)
            for prev_population in hive["PreviousPopulations"]:
                prev_bee_install_date = prev_population["BeeInstallDate"]
                next_bee_install_date = self._get_next_bee_install_date(
                    hive_document=hive, current_population_marker=prev_population["PopulationMarker"]
                )
                if prev_bee_install_date <= end_date and start_date <= next_bee_install_date:
                    hive_name_with_population_marker = (f"{hive_name}{prev_population['PopulationMarker']}"
                                                        if prev_population["PopulationMarker"] != 'A' else hive_name)
                    active_hives.append(hive_name_with_population_marker)
        return active_hives

    @staticmethod
    def _retrieve_middle_frame_from_video(video_filepath: str, frame_number: int) -> Optional[np.ndarray]:
        """
        Retrieves a random frame from the video at the given file path. The frame number is also returned.
        The function assumes that the video may be h264 and as such loops through the video to find the
        desired frame rather than setting it outright.

        Args:
            video_filepath (str): The file path to the video.

        Returns:
            Optional[np.ndarray]: A tuple containing the frame and the frame number.
        """
        cap = cv2.VideoCapture(video_filepath)
        logger.debug(f'Retrieving frame {frame_number} from video at {video_filepath}.')
        i = 0
        frame = None
        file_extension = video_filepath.split('.')[-1]
        if file_extension == 'h264':
            while i < frame_number:
                ret, frame = cap.read()
                if frame is None:
                    logger.warning(f"Could not read frame {i} from video at {video_filepath}. Returning None.")
                    break
                i += 1
        elif file_extension == 'mp4':
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if frame is None:
                logger.warning(f"Could not read frame {frame_number} from video at {video_filepath}. Returning None.")
        else:
            logger.warning(f"Could not read video at {video_filepath}. Currently only supports .h264 and .mp4, "
                           f"got .{file_extension}. Returning None.")
        return frame


    @staticmethod
    def _get_next_bee_install_date(hive_document: dict, current_population_marker: str) -> datetime:
        """
        Convenience method to abstract finding the next bee install date from a hive document.

        Args:
            hive_document (dict): The hive document to find the next bee install date for.
            current_population_marker (str): The current population marker for the hive, e.g. 'A', 'B', 'C', etc.
              This method looks for the next population marker in the hive document, and returns the bee install date.

        Returns:
            datetime: The next bee install date.
        """
        desired_population_marker_ord = ord(current_population_marker) + 1  # Getting the next population marker
        if desired_population_marker_ord == ord(hive_document["PopulationMarker"]):
            next_bee_install_date = hive_document["BeeInstallDate"]
        else:
            for prev_population in hive_document["PreviousPopulations"]:
                if desired_population_marker_ord == ord(prev_population["PopulationMarker"]):
                    next_bee_install_date = prev_population["BeeInstallDate"]
                    break
            else:
                raise ValueError(f"Could not find the next bee install date for hive {hive_document['HiveName']}."
                                 f" (Population {hive_document['PopulationMarker']} has no successor.)")
        return next_bee_install_date


if __name__ == '__main__':
    auth_json = os.path.abspath("auth.json")
    with open(auth_json, "r") as f:
        auth = json.load(f)
    mongo_client = MongoHelper.connect_to_remote_client(username=auth["username"], password=auth["password"])
    pretrained_weights_path = os.path.abspath("trained_models/final_model.pt")
    yolo_model = YOLOModel(pretrained_weights_path=pretrained_weights_path, mongo_client=mongo_client)
    start_date = datetime(2023, 8, 1)
    end_date = datetime(2023, 11, 1)
    start_time = time(15, 0, 0)
    end_time = time(15, 0, 0)
    hive_list = yolo_model.get_active_hives_in_time_frame(start_date=start_date, end_date=end_date)
    # hive_list = ["AppMAIS11L", "AppMAIS11R", "AppMAIS6L", "AppMAIS6R", "AppMAIS12L", "AppMAIS12R"]
    # hive_list = ["AppMAIS8L", "AppMAIS8R", "AppMAIS13L", "AppMAIS13R"]
    results = yolo_model.run_model_on_videos(
        start_date=start_date, end_date=end_date, hive_list=hive_list, start_time=start_time, end_time=end_time
    )
    # graph the results
    YOLOModel.graph_model_results_for_hive(
        results=results, hive_names=hive_list, alpha=0.9, figsize=(20, 20), font_size=22, markersize=11
    )

