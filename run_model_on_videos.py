from typing import Optional, List, Tuple

import ultralytics
import os
from datetime import datetime, time, timedelta
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
                 batch_size: Optional[int] = 64, desired_frame_ind: Optional[int] = 897):
        logger.info(f"Initializing YOLO model with pretrained weights at {pretrained_weights_path}.")
        self._model = ultralytics.YOLO(model=pretrained_weights_path)
        self._mongo_client = mongo_client
        self._bee_db = self._mongo_client.beeDB
        self._video_files_collection = self._bee_db.VideoFiles
        self._hive_collection = self._bee_db.HiveWorkspace
        self._drone_worker_count_collection = self._bee_db.DroneWorkerCount
        self._confidence_threshold = confidence_threshold
        self._batch_size = batch_size
        self._desired_frame_ind = desired_frame_ind


    def run_model_on_videos(self, start_datetime: datetime, end_datetime: datetime, hive_list: Optional[List[str]] = None,
                            start_time: Optional[time] = None, end_time: Optional[time] = None,
                            upload_to_mongo: Optional[bool] = True, stride: Optional[int] = 1,
                            exclude_months: Optional[List[int]] = None) -> (
            pd.DataFrame):
        """
        Retrieves random frames from all videos listed in the date range for the given hives, and runs the YOLO model
        on those frames. Number of drones and workers are returned.
        Args:
            start_datetime (datetime): The start date for the videos.
            end_datetime (datetime): The end date for the videos.
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
            exclude_months (Optional[List[int]]): A list of months to exclude from the query. If None, all months will
                be used.

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
            hive_list = self.get_active_hives_in_time_frame(start_date=start_datetime, end_date=end_datetime)
        logger.info(f"Running YOLO model on videos for hives {hive_list} between {start_datetime} and {end_datetime}.")
        results = []
        already_processed_frames = []
        for hive in hive_list:
            hive_name, population_marker = self.get_population_marker_from_hive_name(hive)
            video_filepaths = self.get_video_filepaths_for_hive(
                hivename=hive, start_date=start_datetime, end_date=end_datetime, start_time=start_time, end_time=end_time,
                stride=stride, exclude_months=exclude_months
            )
            already_processed_frames_for_hive, video_filepaths = self._find_previously_processed_frames(
                video_filepaths=video_filepaths
            )
            already_processed_frames += already_processed_frames_for_hive
            frame_related_data = []
            frames = []
            for video_filepath in video_filepaths:
                date_str, time_str = video_filepath.split("@")[1], video_filepath.split("@")[2].split('.')[0]
                timestamp = datetime.strptime(f"{date_str}@{time_str}", "%Y-%m-%d@%H-%M-%S")
                frame = self._retrieve_frame_from_video(video_filepath, self._desired_frame_ind)
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
    def plot_model_results(results: pd.DataFrame, hive_names: List[str],
                           alpha: Optional[float]=0.7, figsize: Optional[Tuple[int, int]] = (20, 20),
                           font_size: Optional[int] = 22, markersize: Optional[int] = 10, tickwidth: Optional[int] = 3,
                           metric: Optional[str] = "DroneToWorkerRatio", ylabel: Optional[str] = "Drone to Worker Ratio",
                           plot_title: Optional[str] = "Number of Drones and Workers Detected",
                           plot_rolling_average: Optional[bool] = False, moving_rolling_window: Optional[int] = 5,
                           only_show_rolling_average: Optional[bool] = False):
        """
        Plots the number of drones and workers detected in each frame.

        Args:
            results (pd.DataFrame): The DataFrame containing the results of the model.
            hive_names (List[str]): The names of the hives to plot. These names should include the population marker
              (for the time being).
            alpha (Optional[float]): The alpha value for the scatter plot. Default is 0.7.
            figsize (Optional[Tuple[int, int]]): The size of the figure. Default is (20, 20).
            font_size (Optional[int]): The font size for the plot. Default is 22.
            markersize (Optional[int]): The marker size for the plot. Default is 10.
            tickwidth (Optional[int]): The size of the ticks on the plot. Default is 22.
            metric (Optional[str]): The metric to plot. Default is "DroneToWorkerRatio".
            ylabel (Optional[str]): The y-axis label of the plot. Default is "Drone to Worker Ratio". X axis is always
              the time stamp.
            plot_title (Optional[str]): The title of the plot. Default is "Number of Drones and Workers Detected".
            plot_rolling_average (Optional[bool]): Whether to plot the rolling average. Default is False.
            moving_rolling_window (Optional[int]): The window size for the rolling average. Default is 5, ignored
              if plot_moving_average is False.
            only_show_rolling_average (Optional[bool]): Whether to only show the rolling average. Default is False.
        """
        # Make sure there is at least one entry in the DataFrame
        if results.shape[0] == 0:
            logger.warning("No results found in the DataFrame. Exiting.")
            return
        # Set the plot parameters
        plt.rcParams.update({'font.size': font_size})
        plt.rcParams.update({'lines.markersize': markersize})
        fig, ax = plt.subplots(figsize=figsize)
        plt.tick_params(width=tickwidth, length=tickwidth * 3)
        plt.rcParams['lines.linewidth'] = tickwidth

        # sort the results by timestamp
        results = results.sort_values(by="TimeStamp")
        for hive_name in hive_names:
            hive_name, population_marker = YOLOModel.get_population_marker_from_hive_name(hive_name)
            hive_results = results[results["HiveName"] == hive_name]
            hive_results = hive_results[hive_results["PopulationMarker"] == population_marker]
            hive_label = f"{hive_name}{population_marker}" if population_marker != 'A' else hive_name
            if not only_show_rolling_average:
                ax.scatter(hive_results["TimeStamp"], hive_results[metric],
                           label=hive_label, alpha=alpha)
            if plot_rolling_average:
                moving_average = hive_results[metric].rolling(window=moving_rolling_window).mean()
                ax.plot(hive_results["TimeStamp"], moving_average, label=f"{hive_label} Rolling Average",
                        linestyle='dashed')
        ax.set_xlabel("Time Stamp")
        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45)
        ax.set_ylabel(ylabel)
        ax.set_title(plot_title)
        # Add a legend on the outside of the plot
        ncols = 1 if len(hive_names) < 10 else 2
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=ncols)
        plt.tight_layout()
        plt.show()

    def find_consecutive_ratios_over(self, drone_to_worker_threshold: float, start_date: datetime, end_date: datetime,
                                     consecutive_days_threshold: int, hive_list: Optional[List[str]] = None) -> (
            List[Tuple[str, datetime, datetime]]):
        """
        Finds the consecutive days when the drone to worker ratio is over the given threshold.
        Args:
            drone_to_worker_threshold (float): The threshold for the drone to worker ratio.
            start_date (datetime):
            end_date (datetime):
            consecutive_days_threshold (int): Number of consecutive days to look for.

        Returns:
            List[Tuple[str, datetime, datetime]]: A list of tuples containing the hive name, the start date of the
              consecutive days, and the end date of the consecutive days.
        """
        logger.info(f'Finding all sets of {consecutive_days_threshold} consecutive days where the drone to worker ratio is over '
                    f'{drone_to_worker_threshold}.')
        if hive_list is None:
            hive_list = self.get_active_hives_in_time_frame(start_date=start_date, end_date=end_date)
        results = []
        for hive in hive_list:
            hive_name, population_marker = self.get_population_marker_from_hive_name(hive)
            hive_results = self._drone_worker_count_collection.find({
                "HiveName": hive_name,
                "PopulationMarker": population_marker,
                "TimeStamp": { "$gte": start_date, "$lte": end_date },
                "DroneToWorkerRatio": { "$gte": drone_to_worker_threshold }
            })
            hive_results_list = []
            for result in hive_results:
                hive_results_list.append(result)
            # sort by timestamp
            hive_results_list = sorted(hive_results_list, key=lambda x: x["TimeStamp"])
            consecutive_days = 0
            skipped_entries = 0
            met_threshold = False
            for i in range(len(hive_results_list) - 1):
                # if the videos are one day apart (ignoring time, just date)
                if hive_results_list[i + 1]["TimeStamp"].day - hive_results_list[i]["TimeStamp"].day == 1:
                    consecutive_days += 1
                # otherwise, if the videos are the same day
                elif hive_results_list[i + 1]["TimeStamp"].day - hive_results_list[i]["TimeStamp"].day == 0:
                    skipped_entries += 1
                    continue
                else:
                    if met_threshold:
                        results.append(
                            (hive_name, hive_results_list[i - (consecutive_days + skipped_entries)]["TimeStamp"],
                             hive_results_list[i]["TimeStamp"])
                        )
                        met_threshold = False
                    consecutive_days = 0
                    skipped_entries = 0
                if consecutive_days == consecutive_days_threshold:
                    # We've met the threshold for the number of consecutive days we're looking for, but there may be
                    # more consecutive days after this one.
                    met_threshold = True
            # If we're at the end of the list, we need to check if we've met the threshold (if met_threshold is
            # still True, that means we haven't added the last set of consecutive days to the results list).
            if met_threshold:
                results.append(
                    (hive_name, hive_results_list[len(hive_results_list) - (consecutive_days + skipped_entries + 1)]["TimeStamp"],
                     hive_results_list[len(hive_results_list) - 1]["TimeStamp"])
                )
        return results


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

    def _find_previously_processed_frames(self, video_filepaths) -> Tuple[List[dict], List[str]]:
        """
        Finds the frames that have already been processed for the given set of video filepaths. Uses
        ``self._desired_frame_ind`` to find the frame number to check for.
        Args:
            video_filepaths: The filepaths for the videos to check for processed frames.

        Returns:
            Tuple[List[dict], List[str]]: A list of dictionaries containing the processed frame data, and a list of
              filepaths from ``video_filepaths`` that were not found in the database.
        """
        # First have to check for filepaths with either extension since the database may have been updated
        original_filepaths = video_filepaths
        video_filepaths_with_h264_extension = [filepath.replace("h264", "mp4") for filepath in video_filepaths]
        video_filepaths_with_mp4_extension = [filepath.replace("mp4", "h264") for filepath in video_filepaths]
        video_filepaths = video_filepaths_with_h264_extension + video_filepaths_with_mp4_extension
        query = {
            "FrameNumber": self._desired_frame_ind,
            "FilePath": { "$in": video_filepaths }
        }
        results = []
        processed_frames_cursor = self._drone_worker_count_collection.find(query)
        for document in processed_frames_cursor:
            try:
                # Have to remove both copies of the filepath from the list
                video_filepaths.remove(document["FilePath"].replace("h264", "mp4"))
                video_filepaths.remove(document["FilePath"].replace("mp4", "h264"))
            except ValueError:
                logger.warning(f"Could not remove {document['FilePath']} from video_filepaths. Possibly a duplicate "
                               f"record? Skipping...")
                continue
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
        # Lastly, have to remove the non-original filepaths from the list:
        video_filepaths = [filepath for filepath in video_filepaths if filepath in original_filepaths]
        return results, video_filepaths


    def get_video_filepaths_for_hive(self, hivename: str, num_videos: Optional[int] = None,
                                     start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                                     start_time: Optional[time] = None, end_time: Optional[time] = None,
                                     stride: Optional[int] = 1, exclude_months: Optional[List[int]] = None) -> (
            List[str]):
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
            exclude_months (Optional[List[int]]): A list of months to exclude from the query. If None, all months will
              be used.

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
            if exclude_months is not None:
                video_month = video_document["TimeStamp"].month
                if video_month in exclude_months:
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

    def get_model_output_for_frame_from_video(self, video_filepath: str, frame_number: int):
        """
        Retrieves the output of the model for the given frame number in the video at the given file path.

        Args:
            video_filepath (str): The file path to the video.
            frame_number (int): The frame number to get the model output for.

        Returns:
            ultralytics.YOLO: The output of the model for the given frame.
        """
        frame = self._retrieve_frame_from_video(video_filepath, frame_number)
        if frame is None:
            logger.error(f"Could not retrieve frame {frame_number} from video at {video_filepath}. "
                         f"Is the filepath correct?")
            return None
        predictions = self._model.predict(frame, conf=self._confidence_threshold, save=False, stream=True)
        edited_frame = frame.copy()
        num_drones = 0
        num_workers = 0
        for j, prediction in enumerate(predictions):
                for box in prediction.boxes.cpu():
                    x1, y1, x2, y2, conf, class_id = box.data.tolist()[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    color = (0, 0, 0)
                    if class_id == DetectionClasses.DRONE.value:
                        color = (0, 0, 255)
                        num_drones += 1
                    elif class_id == DetectionClasses.WORKER.value:
                        color = (140, 230, 240)
                        num_workers += 1
                    cv2.rectangle(edited_frame, (x1, y1), (x2, y2), color, 2)
        logger.info(f"Detected {num_drones} drones and {num_workers} workers in frame {frame_number} of video at "
                    f"{video_filepath}.")
        return edited_frame

    @staticmethod
    def _retrieve_frame_from_video(video_filepath: str, frame_number: int) -> Optional[np.ndarray]:
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
        frame = None
        file_extension = video_filepath.split('.')[-1]
        if file_extension == 'h264':
            i = 1  # Using 1-indexing so that the frame matches the frame index for mp4 videos
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


class DataType(Enum):
    """
    An enumeration of the different types of data that can be retrieved from the database.
    The value should be the desired key for the data in the database.
    """
    SCALE = "Scale"
    TEMPERATURE = "Temperature"
    HUMIDITY = "Humidity"
    VIDEO = "FileSize"


class OtherDataGetter:
    """
    Gets other data from the database, other than the data retrieved by the YOLOmodel class.
    """

    def __init__(self, mongo_client: MongoClient):
        self._bee_db = mongo_client.beeDB
        self._hive_collection = self._bee_db.HiveWorkspace
        self._video_files_collection = self._bee_db.VideoFiles
        self._scale_collection = self._bee_db.Scale
        self._temperature_humidity_collection = self._bee_db.TemperatureHumidity

    def get_data_for_hive_range(self, hive_names: List[str], data_type: DataType,
                                start_date: datetime, end_date: datetime, start_time: Optional[time] = None,
                                end_time: Optional[time] = None):
        """
        Gets data for the given hive in the given time frame
        Args:
            hive_names:
            data_type:
            start_date:
            end_date:
            start_time:
            end_time:

        Returns:

        """
        if data_type == DataType.SCALE:
            collection = self._scale_collection
        elif data_type == DataType.TEMPERATURE or data_type == DataType.HUMIDITY:
            collection = self._temperature_humidity_collection
        elif data_type == DataType.VIDEO:
            collection = self._video_files_collection
        else:
            raise ValueError(f"Data type {data_type} not recognized.")
        query = {
            "HiveName": { "$in": hive_names},
            "TimeStamp": { "$gte": start_date, "$lte": end_date }
        }
        cursor = collection.find(query)
        results = []
        for document in cursor:
            if start_time is not None or end_time is not None:
                document_time = document["TimeStamp"].time()
                if start_time is not None and document_time < start_time:
                    continue
                if end_time is not None and document_time > end_time:
                    continue
            if document[data_type.value] == '':
                continue
            hive_name, population_marker = YOLOModel.get_population_marker_from_hive_name(document["HiveName"])
            results.append({"TimeStamp": document["TimeStamp"], f"{data_type.value}": document[data_type.value],
                           "HiveName": hive_name, "PopulationMarker": population_marker})
        return pd.DataFrame(results)




if __name__ == '__main__':
    auth_json = os.path.abspath("auth.json")
    with open(auth_json, "r") as f:
        auth = json.load(f)
    mongo_client = MongoHelper.connect_to_remote_client(username=auth["username"], password=auth["password"])
    pretrained_weights_path = os.path.abspath("trained_models/final_model.pt")
    desired_frame_index = 1794 // 2 # Halfway through the video
    yolo_model = YOLOModel(
        pretrained_weights_path=pretrained_weights_path, mongo_client=mongo_client, confidence_threshold=0.64,
        batch_size=64, desired_frame_ind=desired_frame_index
    )
    start_date = datetime(2022, 6, 20)
    end_date = datetime(2022, 7, 10)

    # start_time = end_time = time(15, 0, 0)
    start_time = time(12, 0, 0)
    end_time = time(16, 0, 0)
    # hive_list = yolo_model.get_active_hives_in_time_frame(start_date=start_date, end_date=end_date)
    hive_list = ["AppMAIS12L", "AppMAIS12R"]
    results = yolo_model.run_model_on_videos(
        start_datetime=start_date, end_datetime=end_date, hive_list=hive_list, start_time=start_time, end_time=end_time,
        upload_to_mongo=True, stride=1, # exclude_months=[12, 1, 2, 3]
    )
    # graph the results
    YOLOModel.plot_model_results(
        results=results, hive_names=hive_list, alpha=0.9, figsize=(20, 20), font_size=22, markersize=11, tickwidth=4,
        # plot_rolling_average=True, moving_rolling_window=(end_time.hour - start_time.hour) * 60 // 5 * 4,
        # only_show_rolling_average=True,
        # metric="DroneToWorkerRatio", ylabel="Drone to Worker Ratio", plot_title="Drone to Worker Ratio Against Time"
        # metric="NumDrones", ylabel="Number of Drones Detected", plot_title="Number of Drones Detected Against Time"
        metric="NumWorkers", ylabel="Number of Workers Detected", plot_title="Number of Workers Detected Against Time"
    )

    scale_results = OtherDataGetter(mongo_client=mongo_client).get_data_for_hive_range(
        hive_names=hive_list, data_type=DataType.SCALE, start_date=start_date, end_date=end_date
    )
    YOLOModel.plot_model_results(
        scale_results, hive_names=hive_list, alpha=0.9, figsize=(20, 20), font_size=22, markersize=11, tickwidth=4,
        metric="Scale", ylabel="Scale (kg)", plot_title="Scale Against Time"
    )

    # results = yolo_model.find_consecutive_ratios_over(
    #     drone_to_worker_threshold=0.5, start_date=start_date, end_date=end_date, consecutive_days_threshold=4
    # )
    # print(results)

    # video_filepath = "/mnt/appmais/AppMAIS13R/2023-07-15/video/AppMAIS13R@2023-07-15@15-00-00.mp4"
    # prediction = yolo_model.get_model_output_for_frame_from_video(video_filepath, desired_frame_index + 1)
    # if prediction is not None:
    #     plt.imshow(cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB))
    #     plt.show()
