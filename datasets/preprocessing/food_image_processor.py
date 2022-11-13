import os
from typing import List, Tuple

import yaml

from constants import IMAGE_DIR
from datasets.food_images_dataset import FoodImagesDataset
from datasets.preprocessing.base_processor import BaseProcessor, ProcessingPaths, ProcessingSettings


class FoodImageProcessor(BaseProcessor):

    def __init__(self):
        """
        Handles processing for the food image dataset
        """
        super().__init__(FoodImagesDataset.DATASET_PATH_CREATOR, IMAGE_DIR)

    def create_output_paths(self, path_to_food_category: str) -> ProcessingPaths:
        """
        Creates the output paths for each input image dir
        :param path_to_food_category: path to input images
        :return: a list of tuple containing the input and output image path pairs
        """
        output_paths: List[Tuple[str, str]] = []
        food_category = os.path.split(path_to_food_category)[1]

        for image_name in os.listdir(path_to_food_category):
            if image_name[0] == ".":
                continue
            input_path = os.path.join(path_to_food_category, image_name)
            output_directory_path = os.path.join(self.dataset_path_creator.image_dir, food_category)
            if not os.path.isdir(output_directory_path):
                os.mkdir(output_directory_path)
            output_image_path = os.path.join(output_directory_path, image_name)
            output_paths.append((input_path, output_image_path))
        return output_paths

    @staticmethod
    def _list_dir(path_to_dir: str) -> List[str]:
        """
        Gets a list of all files/dirs in the directory
        :param path_to_dir: the path to the directory
        :return: a list of all files/dirs in the directory
        """
        return list(filter(lambda f: f[0] != ".", os.listdir(path_to_dir)))

    def post_process(self, settings: ProcessingSettings):
        """
        Handles making the label file after all pictures have been processed
        :param settings: contains the appropriate settings for the processing run
        :return: None
        """
        data = {}
        for image_class in self._list_dir(self.dataset_path_creator.image_dir):
            image_class_path = os.path.join(self.dataset_path_creator.image_dir, image_class)
            for image_name in self._list_dir(image_class_path):
                if image_name in data:
                    print("Collision at: ", image_name)
                image_id = os.path.split(image_name)[0]
                data[image_id] = image_class
        self.write_yaml(data, self.dataset_path_creator.label_file)

    @staticmethod
    def write_yaml(data: dict, path_to_file: str):
        """
        Writes to the yaml file
        :param data: a dictionary of the data
        :param path_to_file: path to save the file to
        :return: None
        """
        with open(path_to_file, 'w') as f:
            yaml.dump(data, f)
