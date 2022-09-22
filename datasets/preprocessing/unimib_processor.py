import os

from datasets.preprocessing.abstract_processor import AbstractProcessor, ProcessingPaths
from constants import PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT, IMAGE_DIR
from datasets.unimib_dataset import UnimibDataset


class UnimibProcessor(AbstractProcessor):
    PATH_TO_UNIMIB2016 = os.path.join(PATH_TO_PROJECT, UnimibDataset.DIR_NAME.upper())
    PATH_TO_UNIMIB2016_INPUT = os.path.join(PATH_TO_UNIMIB2016, "UNIMIB2016-images")
    PATH_TO_UNIMIB2016_OUTPUT = os.path.join(PATH_TO_OUTPUT_DIR, UnimibDataset.DIR_NAME, IMAGE_DIR)

    def __init__(self):
        super().__init__(self.PATH_TO_UNIMIB2016_INPUT)

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        return self.create_generic_single_output(entry_name, self.PATH_TO_UNIMIB2016_OUTPUT)
