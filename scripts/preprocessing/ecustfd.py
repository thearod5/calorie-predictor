import os

from scripts.preprocessing.processor import ImageFolderProcessor, ProcessingPaths, \
    create_generic_single_output
from scripts.preprocessing.runner import PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT


class EcustfdProcessor(ImageFolderProcessor):
    ECUSTFD_NAME = "eucstfd"
    PATH_TO_ECUSTFD = os.path.join(PATH_TO_PROJECT, ECUSTFD_NAME)
    PATH_TO_ECUSTFD_INPUT = os.path.join(PATH_TO_ECUSTFD, "JPEGImages")
    PATH_TO_ECUSTFD_OUTPUT = os.path.join(PATH_TO_OUTPUT_DIR, ECUSTFD_NAME, "images")

    def __init__(self):
        super().__init__(self.PATH_TO_ECUSTFD_INPUT)

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        return create_generic_single_output(entry_name, self.PATH_TO_ECUSTFD_OUTPUT)
