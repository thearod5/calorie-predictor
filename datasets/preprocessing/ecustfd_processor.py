import os

from datasets.abstract_dataset import DatasetPathCreator
from datasets.preprocessing.base_processor import BaseProcessor, ProcessingPaths
from constants import PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT, IMAGE_DIR
from datasets.eucstfd_dataset import EucstfdDataset


class EcustfdProcessor(BaseProcessor):

    def __init__(self):
        """
        Handles processing for the eucstfd dataset
        """
        super().__init__(EucstfdDataset.dataset_paths_creator, "JPEGImages")

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        """
        Creates the path to the output for each image
        :param entry_name: the name of the image
        :return: the paths
        """
        return self.create_generic_single_output(entry_name, self.output_dataset_path_creator.image_dir)
