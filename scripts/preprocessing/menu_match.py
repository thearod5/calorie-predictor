import os

from scripts.preprocessing.processor import ImageFolderProcessor, PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT, \
    ProcessingPaths, \
    create_generic_single_output


class MenuMatchPrecessor(ImageFolderProcessor):
    PATH_TO_MENU_MATCH = os.path.join(PATH_TO_PROJECT, "menu_match_dataset")
    PATH_TO_MENU_MATCH_INPUT = os.path.join(PATH_TO_MENU_MATCH, "foodimages")
    PATH_TO_MENU_MATCH_OUTPUT = os.path.join(PATH_TO_OUTPUT_DIR, "menu_match", "images")

    def __init__(self):
        super().__init__(self.PATH_TO_MENU_MATCH_INPUT)

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        return create_generic_single_output(entry_name, self.PATH_TO_MENU_MATCH_OUTPUT)
