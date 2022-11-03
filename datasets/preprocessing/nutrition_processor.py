import csv
import os
from typing import List, Tuple

from constants import IMAGE_DIR, IMAGE_NAME_SEPARATOR, PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT
from datasets.nutrition_dataset import NutritionDataset
from datasets.preprocessing.base_processor import BaseProcessor, ProcessingPaths, ProcessingSettings


class NutritionJpgImagesProcessor(BaseProcessor):
    INPUT_IMAGE_NAME = "rgb.png"
    IMAGE_SUFFIX = "overhead"

    def __init__(self):
        super().__init__(NutritionDataset.dataset_paths_creator, "realsense_overhead")

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        """
        Creates the path to the output for each image
        :param entry_name: the name of the image
        :return: the paths
        """
        dish_name = os.path.split(entry_name)[-1]
        new_image_name = IMAGE_NAME_SEPARATOR.join([dish_name, self.IMAGE_SUFFIX + ".jpg"])
        output_image_path = os.path.join(NutritionDataset.dataset_paths_creator.image_dir, new_image_name)
        input_path = os.path.join(entry_name, self.INPUT_IMAGE_NAME)
        return [(input_path, output_image_path)]


class NutritionH264ImagesProcessor(BaseProcessor):
    INPUT_IMAGE_NAME = "rgb.png"
    IMAGE_SUFFIX = "overhead"
    H264_NAME_TEMPLATE = "camera_%s.h264"

    def __init__(self):
        super().__init__(NutritionDataset.dataset_paths_creator, "side_angles")

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        """
        Creates the path to the output for each image
        :param entry_name: the name of the image
        :return: the paths
        """
        output_paths: List[Tuple[str, str]] = []
        dish_name = os.path.split(entry_name)[-1]
        h264_image_names = ["A", "B", "C", "D"]
        for h264_image_id in h264_image_names:
            input_image_name = self.H264_NAME_TEMPLATE % h264_image_id
            new_image_name = IMAGE_NAME_SEPARATOR.join([dish_name, h264_image_id + ".jpg"])
            output_image_path = os.path.join(sutritionDataset.dataset_paths_creator.image_dir, new_image_name)
            input_image_path = os.path.join(entry_name, input_image_name)
            output_paths.append((input_image_path, output_image_path))
        return output_paths


class NutritionProcessor(BaseProcessor):

    ORIG_METADATA_FILES = ['dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv']
    LABEL2REMOVE = "ingr_"

    def __init__(self):
        super().__init__(NutritionDataset.dataset_paths_creator, "imagery")

    def pre_process(self, settings: ProcessingSettings):
        """
        Creates the metadata csv files for the nutrition dataset
        :param settings: contains the appropriate settings for the processing run
        :return: None
        """
        for filename in enumerate(NutritionDataset.DATA_FILENAMES):
            self._create_new_metadata_csv(filename)

    def process(self, settings: ProcessingSettings) -> None:
        """
        Overrides the base processor to ensure processing of both Jpg and H264 images
        :param settings: contains the appropriate settings for the processing run
        :return: None
        """
        NutritionJpgImagesProcessor().process(settings)
        NutritionH264ImagesProcessor().process(settings)

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        """
        Not used
        """
        pass

    def _create_new_metadata_csv(self, filename: str):
        """
        Creates the metadata csv file from the original
        :param filename: the name of the file to create the metadata file from
        :return: None
        """
        new_data_filepath = os.path.join(self.dataset_path_creator.dataset_dir, filename)
        with open(new_data_filepath, "w") as new_file:
            writer = csv.writer(new_file)
            orig_data_filepath = os.path.join(self.dataset_path_creator.source_dir, filename)
            with open(orig_data_filepath, newline='') as orig_file:
                reader = csv.reader(orig_file)
                for row in reader:
                    new_row = [item for item in row if self.LABEL2REMOVE not in item]
                    writer.writerow(new_row)
