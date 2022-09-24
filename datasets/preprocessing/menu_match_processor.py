import os

from datasets.preprocessing.base_processor import BaseProcessor, ProcessingPaths
from datasets.preprocessing.runner import PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT, IMAGES_DIR
from datasets.menu_match_dataset import MenuMatchDataset
import pandas as pd
import yaml


class MenuMatchPrecessor(BaseProcessor):
    PATH_TO_MENU_MATCH = os.path.join(PATH_TO_PROJECT, "menu_match_dataset")
    PATH_TO_MENU_MATCH_INPUT = os.path.join(PATH_TO_MENU_MATCH, "foodimages")
    PATH_TO_MENU_MATCH_OUTPUT = os.path.join(PATH_TO_OUTPUT_DIR, MenuMatchDataset.DIR_NAME, IMAGES_DIR)

    ITEMS_INFO_FILE = os.path.join(PATH_TO_MENU_MATCH, 'items_info.txt')
    LABEL_FILE = os.path.join(PATH_TO_MENU_MATCH, 'labels.txt')
    DATA_FILE = os.path.join(PATH_TO_MENU_MATCH, MenuMatchDataset.DATA_FILE)

    MISSING_IMGS = {'img10.jpg'}

    def __init__(self):
        super().__init__(self.PATH_TO_MENU_MATCH_INPUT)

    def pre_process(self):
        if not os.path.exists(self.DATA_FILE):
            image_ingredient_mappings = self.get_image_ingredient_mappings()
            item_info = self._get_item_info()
            self._make_image_calorie_mapping_yml()

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        return self.create_generic_single_output(entry_name, self.PATH_TO_MENU_MATCH_OUTPUT)

    def _get_item_info(self):
        rows = []
        with open(self.ORIG_DATA_FILE, "r") as orig_file:
            for line in orig_file.readlines():
                row = "".join(line.split())
                row = row.replace('\t', '').replace(';', ',')
                rows.append(row.split(';'))
        return DataFrame(rows[1:], columns=row[0])

    @staticmethod
    def _get_image_ingredient_mappings():
        image_ingredient_mappings = {}
        with open(MenuMatchPrecessor.ORIG_LABEL_FILE) as f:
            for line in f.readlines():
                img_data = "".join(line.split()).split(";")
                image_ingredient_mappings[img_data[0]] = img_data[1:-1]
        return image_ingredient_mappings

    @staticmethod
    def _make_image_calorie_mapping_yml(food_info, image_labels_mapping):
        image_calorie_mappings = {}
        for img, labels in image_labels_mapping.items():
            total_calories = 0
            for label in labels:
                total_calories += food_info.loc[label]['Calories']
            image_calorie_mappings[img.split(".")[0]] = int(total_calories)
        with open(self.DATA_FILE, 'w') as file:
            yaml.dump(image_calorie_mappings, file)

    @staticmethod
    def _remove_missing_imgs():
        for missing_img in MenuMatchPrecessor.MISSING_IMGS:
            MenuMatchPrecessor.image_ingredient_mappings.pop(missing_img)  # this image is missing??
