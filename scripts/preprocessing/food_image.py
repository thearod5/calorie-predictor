import os
from typing import List, Tuple

import yaml

from scripts.preprocessing.processor import IMAGE_NAME_SEPARATOR, ImageFolderProcessor, \
    PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT, \
    ProcessingPaths


def write_yaml(data: dict, path_to_file: str):
    with open(path_to_file, 'w') as f:
        yaml.dump(data, f)


class FoodImageProcessor(ImageFolderProcessor):
    FOOD_IMAGES_NAME = "food_images"
    PATH_TO_FOOD_IMAGES = os.path.join(PATH_TO_PROJECT, FOOD_IMAGES_NAME)
    PATH_TO_FOOD_IMAGES_INPUT = os.path.join(PATH_TO_FOOD_IMAGES, "images")
    PATH_TO_FOOD_IMAGES_OUTPUT = os.path.join(PATH_TO_OUTPUT_DIR, FOOD_IMAGES_NAME, "images")
    PATH_TO_LABELS = os.path.join(PATH_TO_OUTPUT_DIR, FOOD_IMAGES_NAME, "labels.yml")

    def __init__(self):
        super().__init__(self.PATH_TO_FOOD_IMAGES_INPUT)

    def create_output_paths(self, path_to_food_category: str) -> ProcessingPaths:
        output_paths: List[Tuple[str, str]] = []
        food_category = os.path.split(path_to_food_category)[1]

        for image_name in os.listdir(path_to_food_category):
            if image_name[0] == ".":
                continue
            image_id = image_name.split(".")[0]
            input_path = os.path.join(path_to_food_category, image_name)
            output_image_name = IMAGE_NAME_SEPARATOR.join([image_id, food_category + ".jpg"])
            output_image_path = os.path.join(self.PATH_TO_FOOD_IMAGES_OUTPUT, output_image_name)
            output_paths.append((input_path, output_image_path))
        return output_paths

    def create_output_file(self):
        image_files = list(filter(lambda f: f[0] != ".", os.listdir(self.PATH_TO_FOOD_IMAGES_OUTPUT)))

        data = {}
        for image_file in image_files:
            image_id, image_class = image_file.split(IMAGE_NAME_SEPARATOR)
            image_class = image_class.split(".")[0]  # remove image extension
            if image_id in data:
                print("Collision at: ", image_id)
            data[image_id] = image_class

        write_yaml(data, self.PATH_TO_LABELS)
