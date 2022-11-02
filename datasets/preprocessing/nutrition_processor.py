import csv
import os
from typing import List, Tuple

from constants import IMAGE_DIR, IMAGE_NAME_SEPARATOR, PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT
from datasets.nutrition_dataset import NutritionDataset
from datasets.preprocessing.base_processor import BaseProcessor, ProcessingPaths, ProcessingSettings


class NutritionJpgImagesProcessor(BaseProcessor):
    INPUT_IMAGE_NAME = "rgb.png"
    IMAGE_SUFFIX = "overhead"

    def __init__(self, path_to_source: str, path_to_output: str):
        super().__init__(os.path.join(path_to_source, "realsense_overhead"))
        self.path_to_output = path_to_output

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        dish_name = os.path.split(entry_name)[-1]
        new_image_name = IMAGE_NAME_SEPARATOR.join([dish_name, self.IMAGE_SUFFIX + ".jpg"])
        output_image_path = os.path.join(self.path_to_output, new_image_name)
        input_path = os.path.join(entry_name, self.INPUT_IMAGE_NAME)
        return [(input_path, output_image_path)]


class NutritionH264ImagesProcessor(BaseProcessor):
    INPUT_IMAGE_NAME = "rgb.png"
    IMAGE_SUFFIX = "overhead"
    H264_NAME_TEMPLATE = "camera_%s.h264"

    def __init__(self, path_to_source: str, path_to_output: str):
        super().__init__(os.path.join(path_to_source, "side_angles"))
        self.path_to_output = path_to_output

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        output_paths: List[Tuple[str, str]] = []
        dish_name = os.path.split(entry_name)[-1]
        h264_image_names = ["A", "B", "C", "D"]
        for h264_image_id in h264_image_names:
            input_image_name = self.H264_NAME_TEMPLATE % h264_image_id
            new_image_name = IMAGE_NAME_SEPARATOR.join([dish_name, h264_image_id + ".jpg"])
            output_image_path = os.path.join(self.path_to_output, new_image_name)
            input_image_path = os.path.join(entry_name, input_image_name)
            output_paths.append((input_image_path, output_image_path))
        return output_paths


class NutritionProcessor(BaseProcessor):
    PATH_TO_SOURCE = os.path.join(PATH_TO_PROJECT, NutritionDataset.DIR_NAME)
    PATH_TO_SOURCE_IMAGES = os.path.join(PATH_TO_SOURCE, "imagery")
    PATH_TO_OUTPUT_IMAGES = os.path.join(PATH_TO_OUTPUT_DIR, NutritionDataset.DIR_NAME, IMAGE_DIR)

    ORIG_METADATA_FILES = ['dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv']

    LABEL2REMOVE = "ingr_"

    def __init__(self):
        super().__init__(self.PATH_TO_OUTPUT_IMAGES)

    def pre_process(self):
        for i, filename in enumerate(NutritionDataset.DATA_FILENAMES):
            new_data_filepath = os.path.join(self.PATH_TO_SOURCE, filename)
            if not os.path.exists(new_data_filepath):
                self._create_new_metadata_csv(new_data_filepath, i)

    def process(self, settings: ProcessingSettings) -> None:
        NutritionJpgImagesProcessor(self.PATH_TO_SOURCE_IMAGES, self.PATH_TO_OUTPUT_IMAGES).process(settings)
        NutritionH264ImagesProcessor(self.PATH_TO_SOURCE_IMAGES, self.PATH_TO_OUTPUT_IMAGES).process(settings)

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        pass

    def _create_new_metadata_csv(self, new_data_filepath, file_no):
        with open(new_data_filepath, "w") as new_file:
            writer = csv.writer(new_file)
            orig_data_filepath = os.path.join(self.PATH_TO_SOURCE, self.ORIG_METADATA_FILES[file_no])
            with open(orig_data_filepath, newline='') as orig_file:
                reader = csv.reader(orig_file)
                for row in reader:
                    new_row = [item for item in row if self.LABEL2REMOVE not in item]
                    writer.writerow(new_row)
