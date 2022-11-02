import os
from typing import Dict, List

import yaml
from pandas import DataFrame

from constants import IMAGE_DIR, PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT
from datasets.menu_match_dataset import MenuMatchDataset
from datasets.preprocessing.base_processor import BaseProcessor, ProcessingPaths


class MenuMatchPrecessor(BaseProcessor):
    PATH_TO_SOURCE = os.path.join(PATH_TO_PROJECT, "menu_match_dataset")
    PATH_TO_SOURCE_IMAGES = os.path.join(PATH_TO_SOURCE, "foodimages")
    PATH_TO_OUTPUT_IMAGES = os.path.join(PATH_TO_OUTPUT_DIR, MenuMatchDataset.DIR_NAME, IMAGE_DIR)

    ITEMS_INFO_FILE = os.path.join(PATH_TO_SOURCE, 'items_info.txt')
    LABEL_FILE = os.path.join(PATH_TO_SOURCE, 'labels.txt')
    DATA_FILE = os.path.join(PATH_TO_SOURCE, MenuMatchDataset.DATA_FILE)

    MISSING_IMGS = {'img10.jpg'}

    def __init__(self):
        super().__init__(self.PATH_TO_SOURCE_IMAGES)

    def pre_process(self):
        if not os.path.exists(self.DATA_FILE):
            image_ingredient_mappings = self._get_image_ingredient_mappings()
            self._remove_missing_imgs(image_ingredient_mappings)
            item_info = self._get_item_info()
            self._make_image_calorie_mapping_yml(item_info, image_ingredient_mappings)

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        return self.create_generic_single_output(entry_name, self.PATH_TO_OUTPUT_IMAGES)

    def _get_item_info(self):
        rows = []
        with open(self.ITEMS_INFO_FILE, "r") as orig_file:
            for line in orig_file.readlines():
                row = "".join(line.split())
                row = row.replace('\t', '').replace(';', ',')
                rows.append(row.split(','))
        df = DataFrame(rows[1:], columns=rows[0]).set_index('labelindatabase')
        df["Calories"] = df["Calories"].astype(int)
        return df

    @staticmethod
    def _get_image_ingredient_mappings() -> Dict[str, List[str]]:
        image_ingredient_mappings = {}
        with open(MenuMatchPrecessor.LABEL_FILE) as f:
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
        with open(MenuMatchPrecessor.DATA_FILE, 'w') as file:
            yaml.dump(image_calorie_mappings, file)

    @staticmethod
    def _remove_missing_imgs(image_ingredient_mapping: Dict[str, List[str]]):
        for missing_img in MenuMatchPrecessor.MISSING_IMGS:
            image_ingredient_mapping.pop(missing_img)  # this image is missing??
