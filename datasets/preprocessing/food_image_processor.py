import os
from typing import List, Tuple

import yaml

from constants import IMAGE_DIR, PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT
from datasets.food_images_dataset import FoodImagesDataset
from datasets.preprocessing.base_processor import BaseProcessor, ProcessingPaths


class FoodImageProcessor(BaseProcessor):
    PATH_TO_SOURCE = os.path.join(PATH_TO_PROJECT, FoodImagesDataset.DIR_NAME)
    PATH_TO_SOURCE_IMAGES = os.path.join(PATH_TO_SOURCE, IMAGE_DIR)
    PATH_TO_OUTPUT = os.path.join(PATH_TO_OUTPUT_DIR, FoodImagesDataset.DIR_NAME)
    PATH_TO_OUTPUT_IMAGES = os.path.join(PATH_TO_OUTPUT, IMAGE_DIR)
    PATH_TO_LABELS = os.path.join(PATH_TO_OUTPUT_DIR, FoodImagesDataset.DIR_NAME, FoodImagesDataset.DATA_FILENAME)

    def __init__(self):
        super().__init__(self.PATH_TO_SOURCE_IMAGES)

    def create_output_paths(self, path_to_food_category: str) -> ProcessingPaths:
        output_paths: List[Tuple[str, str]] = []
        food_category = os.path.split(path_to_food_category)[1]

        for image_name in os.listdir(path_to_food_category):
            if image_name[0] == ".":
                continue
            input_path = os.path.join(path_to_food_category, image_name)
            output_directory_path = os.path.join(self.PATH_TO_OUTPUT, food_category)
            if not os.path.isdir(output_directory_path):
                os.mkdir(output_directory_path)
            output_image_path = os.path.join(output_directory_path, image_name)
            output_paths.append((input_path, output_image_path))
        return output_paths

    def _list_dir(self, path_to_dir: str):
        return list(filter(lambda f: f[0] != ".", os.listdir(path_to_dir)))

    def post_process(self):
        if os.path.exists(self.PATH_TO_LABELS):
            return
        data = {}
        for image_class in self._list_dir(self.PATH_TO_OUTPUT_IMAGES):
            image_class_path = os.path.join(self.PATH_TO_OUTPUT_IMAGES, image_class)
            for image_name in self._list_dir(image_class_path):
                if image_name in data:
                    print("Collision at: ", image_name)
                image_id = os.path.split(image_name)[0]
                data[image_id] = image_class
        print(self.PATH_TO_LABELS)
        self.write_yaml(data, self.PATH_TO_LABELS)

    @staticmethod
    def write_yaml(data: dict, path_to_file: str):
        with open(path_to_file, 'w') as f:
            yaml.dump(data, f)
