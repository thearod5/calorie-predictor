import os
from typing import List, Tuple

import yaml

from datasets.preprocessing.base_processor import BaseProcessor, ProcessingPaths
from constants import IMAGE_NAME_SEPARATOR, PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT, IMAGE_DIR
from datasets.food_images_dataset import FoodImagesDataset


class FoodImageProcessor(BaseProcessor):
    PATH_TO_FOOD_IMAGES = os.path.join(PATH_TO_PROJECT, FoodImagesDataset.DIR_NAME)
    PATH_TO_FOOD_IMAGES_INPUT = os.path.join(FoodImagesDataset.DIR_NAME, IMAGE_DIR)
    PATH_TO_FOOD_IMAGES_OUTPUT = os.path.join(PATH_TO_OUTPUT_DIR, FoodImagesDataset.DIR_NAME, IMAGE_DIR)
    PATH_TO_LABELS = os.path.join(PATH_TO_OUTPUT_DIR, FOOD_IMAGES_NAME, FoodImagesDataset.DATA_FILENAME)

    def __init__(self):
        super().__init__(self.PATH_TO_FOOD_IMAGES_INPUT)

    def create_output_paths(self, path_to_food_category: str) -> ProcessingPaths:
        output_paths: List[Tuple[str, str]] = []
        food_category = os.path.split(path_to_food_category)[1]

        for image_name in os.listdir(path_to_food_category):
            if image_name[0] == ".":
                continue
            input_path = os.path.join(path_to_food_category, image_name)
            output_directory_path = os.path.join(self.PATH_TO_FOOD_IMAGES_OUTPUT, food_category)
            if not os.path.isdir(output_directory_path):
                os.mkdir(output_directory_path)
            output_image_path = os.path.join(output_directory_path, image_name)
            output_paths.append((input_path, output_image_path))
        return output_paths

    def post_process(self):
        image_files = list(filter(lambda f: f[0] != ".", os.listdir(self.PATH_TO_FOOD_IMAGES_OUTPUT)))

        data = {}
        for image_file in image_files:
            image_id, image_class = image_file.split(IMAGE_NAME_SEPARATOR)
            image_class = image_class.split(".")[0]  # remove image extension
            if image_id in data:
                print("Collision at: ", image_id)
            data[image_id] = image_class
        if not os.path.exists(self.PATH_TO_LABELS):
            self.write_yaml(data, self.PATH_TO_LABELS)

    @staticmethod
    def write_yaml(data: dict, path_to_file: str):
        with open(path_to_file, 'w') as f:
            yaml.dump(data, f)
