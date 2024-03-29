import json
import os
from typing import Dict, List
from urllib.request import urlopen

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import DEFAULT_TURK_RESULTS, get_cam_path
from src.datasets.abstract_dataset import AbstractDataset


class HMapCreator:
    def __init__(self, dataset: AbstractDataset, result_file_names: List[str],
                 results_path: str = DEFAULT_TURK_RESULTS):
        """
        Creates heat maps from the bounding boxes defined by runs of the turk task.
        :param dataset: The dataset whose bounding boxes are read.
        :param result_file_names: List of result file names representing a bounding boxe per image.
        :param results_path: The path to find the results file.
        """
        self.dataset = dataset
        self.dataset_name = dataset.dataset_path_creator.name
        self.result_file_names = result_file_names
        self.n_batches = len(result_file_names)
        self.results_path = results_path

    def save_hmap_batches(self):
        """
        For each batch represented by a result file, write bounding boxes as heat maps.
        :return: None
        """
        export_dir = os.path.join(get_cam_path(), self.dataset_name)
        for batch_index, result_file_name in enumerate(tqdm(self.result_file_names)):
            batch_id = batch_index + 1
            print("Starting batch: %d / %d" % (batch_id, self.n_batches))
            self._save_hmap_batch(result_file_name, batch_id, export_dir)

    def save_avg_hmaps(self):
        """
        For each image in the dataset calculate and save the average heat map.
        :return: None
        """
        dataset_path = os.path.join(get_cam_path(), self.dataset.dataset_path_creator.name)
        for image_name in tqdm(self.dataset.get_image_names(with_extension=True)):
            HMapCreator._save_avg_hmap_for_image(dataset_path, image_name)

    @staticmethod
    def get_image_id_from_result(result: Dict) -> str:
        """
        Returns the id of the image from the url within the MTurk result.
        :param result: The result from the mechanical turk task.
        :return: String representing image id.
        """
        result_image_url = result["Input.image_url"]
        return result_image_url.split("/")[-1]

    def get_image_result(self, image_id: str, result_index: int = 0):
        results_df = self._read_result_file(self.result_file_names[result_index])
        for result_index in range(len(results_df)):
            result_row = results_df.iloc[result_index]
            if self.get_image_id_from_result(result_row) == image_id:
                return result_row
        raise ValueError("Could not find result for:" + image_id)

    def _save_hmap_batch(self, result_file_name: str, batch_id: int, batch_id_path: str) -> None:
        """
        Reads file containing bounding boxes and saves them as heat maps.
        :param result_file_name: The name of the
        :param batch_id:
        :param batch_id_path:
        :return:
        """
        results_df = self._read_result_file(result_file_name)
        batch_id_path = os.path.join(batch_id_path, "batches", str(batch_id))
        if not os.path.exists(batch_id_path):
            os.makedirs(batch_id_path)
        for i in tqdm(range(len(results_df))):
            HMapCreator.create_hmap_for_image(results_df.iloc[i], batch_id_path)
        print("Done!")

    def _read_result_file(self, result_file_name: str) -> pd.DataFrame:
        results_data_path = os.path.join(self.results_path, result_file_name)
        return pd.read_csv(results_data_path)

    @staticmethod
    def create_hmap_for_image(bounding_box_item: Dict, batch_id_path: str, write_image: bool = True) -> None:
        """
        box coordinates returned from your model's predictions
        color is the color of the bounding box you would like & 2 is the thickness of the bounding box
        :param bounding_box_item: The turk entry containing bounding box
        :param batch_id_path: Path to the directory of the batch being processed
        :return:
        """
        result_image_url = bounding_box_item["Input.image_url"]
        bounding_boxes = json.loads(bounding_box_item["Answer.annotatedResult.boundingBoxes"])
        if len(bounding_boxes) == 0:
            print("Missing bounding box: " + bounding_box_item["HITId"])
            return
        result_image_box = bounding_boxes[0]
        input_image = HMapCreator.read_image_from_url(result_image_url)

        (start_x, start_y, end_x, end_y) = HMapCreator.get_bounding_box_coordinates(result_image_box)
        hmap = np.zeros(shape=input_image.shape)
        hmap[start_y: end_y, start_x: end_x] = 1
        hmap = (hmap * 255).astype(np.uint8)

        file_name = result_image_url.split("/")[-1]
        export_path = os.path.join(batch_id_path, file_name)

        if write_image:
            cv2.imwrite(export_path, hmap)
        else:
            return hmap

    @staticmethod
    def _save_avg_hmap_for_image(dataset_path: str, image_file_name: str) -> None:
        """
        Averages hmaps for image and saves to dataset path.
        :param dataset_path: The path to the dataset within cam folder.
        :param image_file_name: The name of the image whose hmaps are being processed.
        :return: None
        """
        batch_path = os.path.join(dataset_path, "batches")
        batch_ids = list(filter(lambda f: f[0] != ".", os.listdir(batch_path)))

        avg_hmap = None
        n_batches = 0
        for batch_id in batch_ids:
            hmap_path = os.path.join(batch_path, batch_id, image_file_name)
            if not os.path.exists(hmap_path):
                continue

            hmap = cv2.imread(hmap_path)
            hmap = hmap / 255.0
            hmap = cv2.blur(hmap, ksize=(250, 250))
            avg_hmap = hmap if avg_hmap is None else avg_hmap + hmap
            n_batches += 1
        if avg_hmap is None:
            print("Did not find any heat maps for the %s." % image_file_name)
        else:
            avg_hmap = (avg_hmap / n_batches) * 255
            export_path = os.path.join(dataset_path, image_file_name)
            cv2.imwrite(export_path, avg_hmap)

    @staticmethod
    def read_image_from_url(image_url: str):
        """
        Downloads and reads image from url.
        :param image_url: The url to the image to read
        :return:
        """
        req = urlopen(image_url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(arr, -1)  # 'Load it as it is'

    @staticmethod
    def get_bounding_box_coordinates(image_box: dict):
        """
        Extracts the coordinates of the bounding box.
        :param image_box: Dictionary representing bounding box entry from turk task.
        :return: Tuple representing the starting x and y coordinates followed by ending x and y coordinates.
        """
        start_x = image_box["left"]
        start_y = image_box["top"]
        end_x = start_x + image_box["width"]
        end_y = start_y + image_box["height"]
        return start_x, start_y, end_x, end_y
