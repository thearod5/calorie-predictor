import os

from scripts.preprocessing.processor import ImageFolderProcessor, PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT, \
    ProcessingPaths, \
    create_generic_single_output


class UNIMIB2016Processor(ImageFolderProcessor):
    PATH_TO_UNIMIB2016 = os.path.join(PATH_TO_PROJECT, "UNIMIB2016")
    PATH_TO_UNIMIB2016_INPUT = os.path.join(PATH_TO_UNIMIB2016, "UNIMIB2016-images")
    PATH_TO_UNIMIB2016_OUTPUT = os.path.join(PATH_TO_OUTPUT_DIR, "unimib2016", "images")

    def __init__(self):
        super().__init__(self.PATH_TO_UNIMIB2016_INPUT)

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        return create_generic_single_output(entry_name, self.PATH_TO_UNIMIB2016_OUTPUT)
