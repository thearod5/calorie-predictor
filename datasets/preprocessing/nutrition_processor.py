import csv
import os
from collections import Set
from typing import List, Tuple, Optional
from abc import ABC
from constants import IMAGE_NAME_SEPARATOR
from datasets.abstract_dataset import DatasetPathCreator
from datasets.nutrition_dataset import NutritionDataset
from datasets.preprocessing.base_processor import BaseProcessor, ProcessingPaths, ProcessingSettings
import pandas as pd


class NutritionSubProcessor(BaseProcessor, ABC):

    def __init__(self, dataset_path_creator: DatasetPathCreator, source_image_dir: str, dish_ids):
        """
        Handles processing of the datasets
        :param dataset_path_creator: handles making the paths to the input data for a dataset
        :param source_image_dir: the directory containing the source (unprocessed) images
        """
        self.dish_ids = dish_ids
        super().__init__(dataset_path_creator, source_image_dir)

    @staticmethod
    def _create_single_path(entry_name: str, input_image_name: str, output_image_suffix: str) -> Tuple[str, str]:
        """
        Creates a single path for the nutrition sub processors
        :param entry_name: path to input images
        :param input_image_name: name of the orig input image name
        :param output_image_suffix: ending for the new output image name
        :return: the single path input, output pair
        """
        dish_name = os.path.split(entry_name)[-1]
        new_image_name = IMAGE_NAME_SEPARATOR.join([dish_name, output_image_suffix + ".jpg"])
        output_image_path = os.path.join(NutritionDataset.dataset_paths_creator.image_dir, new_image_name)
        input_image_path = os.path.join(entry_name, input_image_name)
        return input_image_path, output_image_path

    @staticmethod
    def _get_dish_id_from_path(image_path: str) -> str:
        """
        Extracts the dish_id from the path
        :return: the dish id
        """
        image_name = NutritionDataset.get_name_from_path(image_path)
        return NutritionDataset.get_dish_id_from_image_name(image_name)

    def _get_image_paths(self) -> List:
        """
        Gets all paths pointing to the images in a dataset, while filtering missing images
        :return: a list of paths
        """
        return list(filter(lambda p: self._get_dish_id_from_path(p) in self.dish_ids,
                           super()._get_image_paths()))


class NutritionJpgImagesProcessor(NutritionSubProcessor):
    INPUT_IMAGE_NAME = "rgb.png"
    IMAGE_SUFFIX = "overhead"

    def __init__(self, dataset_path_creator: DatasetPathCreator, dishes: Set):
        super().__init__(dataset_path_creator, "realsense_overhead", dishes)

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        """
        Creates the path to the output for each image
        :param entry_name: the name of the image
        :return: the paths
        """
        return [self._create_single_path(entry_name, self.INPUT_IMAGE_NAME, self.IMAGE_SUFFIX)]


class NutritionH264ImagesProcessor(NutritionSubProcessor):
    INPUT_IMAGE_NAME = "rgb.png"
    IMAGE_SUFFIX = "overhead"
    H264_NAME_TEMPLATE = "camera_%s.h264"

    def __init__(self, dataset_path_creator: DatasetPathCreator, dish_ids: Set):
        super().__init__(dataset_path_creator, "side_angles", dish_ids)

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        """
        Creates the path to the output for each image
        :param entry_name: the name of the image
        :return: the paths
        """
        output_paths: List[Tuple[str, str]] = []
        h264_image_names = ["A", "B", "C", "D"]
        for h264_image_id in h264_image_names:
            input_image_name = self.H264_NAME_TEMPLATE % h264_image_id
            path = self._create_single_path(entry_name, input_image_name, h264_image_id)
            output_paths.append(path)
        return output_paths


class NutritionProcessor(BaseProcessor):
    ORIG_METADATA_FILES = ['dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv']
    METADATA_DIR = "metadata"
    LABEL2REMOVE = "ingr_"

    def __init__(self):
        self.dishes = set()
        super().__init__(NutritionDataset.dataset_paths_creator, "imagery")
        self.dataset_path_creator.source_dir = self.image_dir

    def pre_process(self, settings: ProcessingSettings):
        """
        Creates the metadata csv files for the nutrition dataset
        :param settings: contains the appropriate settings for the processing run
        :return: None
        """
        for filename in NutritionDataset.DATA_FILENAMES:
            self.dishes = self.dishes.union(self._create_new_metadata_csv(filename))

    def process(self, settings: ProcessingSettings) -> Optional[Exception]:
        """
        Overrides the base processor to ensure processing of both Jpg and H264 images
        :param settings: contains the appropriate settings for the processing run
        :return: None
        """
        try:
            self.pre_process(settings)
            NutritionJpgImagesProcessor(self.dataset_path_creator, self.dishes).process(settings)
            NutritionH264ImagesProcessor(self.dataset_path_creator, self.dishes).process(settings)
        except Exception as e:
            self.print_exception(e)
            return e

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        """
        Not used
        """
        pass

    def _create_new_metadata_csv(self, filename: str) -> Set:
        """
        Creates the metadata csv file from the original
        :param filename: the name of the file to create the metadata file from
        :return: a set containing the ids of all dishes in the metadata
        """
        new_data_filepath = os.path.join(self.dataset_path_creator.dataset_dir, filename)
        orig_data_filepath = os.path.join(self.dataset_path_creator.source_dir, self.METADATA_DIR, filename)
        processed_rows = []
        dishes = set()
        with open(orig_data_filepath, newline='') as orig_file:
            reader = csv.reader(orig_file)
            for row in reader:
                processed_row = [item for item in row if self.LABEL2REMOVE not in item]
                processed_rows.append(processed_row)
                dishes.add(processed_row[NutritionDataset.id_index])
        pd.DataFrame(processed_rows[1:], columns=processed_rows[0]).to_csv(new_data_filepath)
        return dishes
