import os
from typing import List, Tuple

from scripts.preprocessing.processor import IMAGE_NAME_SEPARATOR, ImageFolderProcessor, \
    PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT, \
    ProcessingPaths

PATH_TO_NUTRITION5K = os.path.join(PATH_TO_PROJECT, "nutrition5k")
PATH_TO_NUTRITION5K_IMAGES = os.path.join(PATH_TO_NUTRITION5K, "imagery")
PATH_TO_NUTRITION5K_OUTPUT = os.path.join(PATH_TO_OUTPUT_DIR, "nutrition5k", "images")


class Nutrition5kJpgImages(ImageFolderProcessor):
    INPUT_IMAGE_NAME = "rgb.png"
    IMAGE_SUFFIX = "overhead"

    def __init__(self):
        super().__init__(os.path.join(PATH_TO_NUTRITION5K_IMAGES, "realsense_overhead"))

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        dish_name = os.path.split(entry_name)[-1]
        new_image_name = IMAGE_NAME_SEPARATOR.join([dish_name, self.IMAGE_SUFFIX + ".jpg"])
        output_image_path = os.path.join(PATH_TO_NUTRITION5K_OUTPUT, new_image_name)
        input_path = os.path.join(entry_name, self.INPUT_IMAGE_NAME)
        return [(input_path, output_image_path)]


class Nutrition5kH264Images(ImageFolderProcessor):
    INPUT_IMAGE_NAME = "rgb.png"
    IMAGE_SUFFIX = "overhead"
    H264_NAME_TEMPLATE = "camera_%s.h264"

    def __init__(self):
        super().__init__(os.path.join(PATH_TO_NUTRITION5K_IMAGES, "side_angles"))

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        output_paths: List[Tuple[str, str]] = []
        dish_name = os.path.split(entry_name)[-1]
        h264_image_names = ["A", "B", "C", "D"]
        for h264_image_id in h264_image_names:
            input_image_name = self.H264_NAME_TEMPLATE % h264_image_id
            new_image_name = IMAGE_NAME_SEPARATOR.join([dish_name, h264_image_id + ".jpg"])
            output_image_path = os.path.join(PATH_TO_NUTRITION5K_OUTPUT, new_image_name)
            input_image_path = os.path.join(entry_name, input_image_name)
            output_paths.append((input_image_path, output_image_path))
        return output_paths
