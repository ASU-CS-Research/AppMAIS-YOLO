from typing import Optional, List

import ultralytics
import os
from datetime import datetime
from MongoUtils.mongo_helper import MongoHelper
from pymongo import MongoClient
import json
from loguru import logger


class YOLOModel:
    """
    A class to represent a YOLO model. This class is a wrapper around the ultralytics.YOLO class, with the purpose of
    connecting to the database and getting a representation of
    """
    def __init__(self, pretrained_weights_path: str, mongo_client: MongoClient):
        logger.info("Initializing YOLO model with pretrained weights at {pretrained_weights_path}.")
        self._model = ultralytics.YOLO(model=pretrained_weights_path)
        self._mongo_client = mongo_client
        self._bee_db = self._mongo_client.beeDB
        self._video_files_collection = self._bee_db.VideoFiles
        self._hive_collection = self._bee_db.HiveWorkspace

    def predict_video(self, video_filepath: str, conf: float = 0.64, save: bool = False):
        """
        Predicts on the video at the given file path.

        Args:
            video_filepath (str): The file path to the video.
            conf (float): The confidence threshold. Default is 0.64, as that reported the highest F1 score during
              training.
            save (bool): Whether to save the video.

        """
        return self._model.predict(video_filepath, conf=conf, save=save)

    """
    The following methods may not be best for the model class, but leaving them here as they're currently needed.
    """
    def get_video_filepaths_for_hive(self, hivename: str, num_videos: Optional[int] = None,
                                     start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> \
            List[str]:
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

        Returns:
            List[str]: A list of filepaths for the videos associated with the given hive.
        """
        logger.debug(f"Querying for video filepaths for hive {hivename} between {start_date} and {end_date}.")
        query = {
            "HiveName": hivename,
        }
        if start_date is not None or end_date is not None:
            query["TimeStamp"] = {}
        if start_date is not None:
            query["TimeStamp"].update({"$gte": start_date})
        if end_date is not None:
            query["TimeStamp"].update({"$lte": end_date})
        videos_cursor = self._video_files_collection.find(query)
        filepaths = []
        for video_document in videos_cursor:
            filepath = video_document["FilePath"]
            filepath = filepath.replace("/usr/local/bee", "/mnt")
            if os.path.exists(filepath):
                filepaths.append(filepath)
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
                hive_name_with_population_marker = (f"{hive_name}" + f"{hive['PopulationMarker']}"
                                                    if hive["PopulationMarker"] != 'A' else hive_name)
                active_hives.append(hive_name_with_population_marker)
            for prev_population in hive["PreviousPopulations"]:
                prev_bee_install_date = prev_population["BeeInstallDate"]
                next_bee_install_date = self._get_next_bee_install_date(
                    hive_document=hive, current_population_marker=prev_population["PopulationMarker"]
                )
                if prev_bee_install_date <= end_date and start_date <= next_bee_install_date:
                    hive_name_with_population_marker = (f"{hive_name}" + f"{hive['PopulationMarker']}"
                                                        if hive["PopulationMarker"] != 'A' else hive_name)
                    active_hives.append(hive_name_with_population_marker)
        return active_hives

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
    print(yolo_model.get_active_hives_in_time_frame(
        start_date=datetime(2023, 7, 1), end_date=datetime(2023, 7, 2)
    ))
    print(yolo_model.get_video_filepaths_for_hive(
        hivename="AppMAIS1LB", num_videos=5, start_date=datetime(2023, 7, 1),
        end_date=datetime(2023, 7, 2)
    ))
