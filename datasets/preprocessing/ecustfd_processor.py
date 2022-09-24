import os

from datasets.preprocessing.base_processor import BaseProcessor, ProcessingPaths
from constants import PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT, IMAGE_DIR
from datasets.eucstfd_dataset import EucstfdDataset


class EcustfdProcessor(BaseProcessor):
    PATH_TO_ECUSTFD = os.path.join(PATH_TO_PROJECT, EucstfdDataset.DIR_NAME)
    PATH_TO_ECUSTFD_INPUT = os.path.join(PATH_TO_ECUSTFD, "JPEGImages")
    PATH_TO_ECUSTFD_OUPUT = os.path.join(PATH_TO_OUTPUT_DIR, EucstfdDataset.DIR_NAME, IMAGE_DIR)

    def __init__(self):
        super().__init__(self.PATH_TO_ECUSTFD_INPUT)

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        return self.create_generic_single_output(entry_name, self.PATH_TO_ECUSTFD_OUPUT)
