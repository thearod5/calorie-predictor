import os
from typing import Dict, List

import yaml
import pandas as pd
from src.datasets.menu_match_dataset import MenuMatchDataset
from src.preprocessing.base_processor import BaseProcessor, ProcessingPaths, ProcessingSettings


class MenuMatchPrecessor(BaseProcessor):
    ITEMS_INFO_FILE = os.path.join(MenuMatchDataset.DATASET_PATH_CREATOR.source_dir, 'items_info.txt')
    LABEL_FILE = os.path.join(MenuMatchDataset.DATASET_PATH_CREATOR.source_dir, "labels.txt")

    INDEX_COL = 'labelindatabase'
    CALORIES_COL = 'Calories'

    TAB = "\t"
    ORIG_DELIMINATOR = ";"
    EMPTY = ''

    MISSING_IMGS = {'img10.jpg'}

    def __init__(self):
        super().__init__(MenuMatchDataset.DATASET_PATH_CREATOR, "foodimages")

    def pre_process(self, settings: ProcessingSettings):
        """
        Creates the label file for the images
        :param settings: contains the appropriate settings for the processing run
        :return: None
        """
        image_ingredient_mappings = self._process_image_ingredient_mappings()
        self._remove_missing_imgs(image_ingredient_mappings)
        item_info = self._process_item_info()
        self._make_image_calorie_mapping_yml(item_info, image_ingredient_mappings)

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        """
        Creates the path to the output for each image
        :param entry_name: the name of the image
        :return: the paths
        """
        return self.create_generic_single_output(entry_name, self.dataset_path_creator.image_dir)

    @staticmethod
    def _process_item_info() -> pd.DataFrame:
        """
        Processes the item info file and returns the data in a dataframe
        :return: a dataframe containing item info
        """
        rows = []
        with open(MenuMatchPrecessor.ITEMS_INFO_FILE, "r") as orig_file:
            for i, line in enumerate(orig_file.readlines()):
                row = MenuMatchPrecessor._process_file_line(line)
                rows.append(row)

        df = pd.DataFrame(rows[1:], columns=rows[0]).set_index(MenuMatchPrecessor.INDEX_COL)
        df[MenuMatchPrecessor.CALORIES_COL] = df[MenuMatchPrecessor.CALORIES_COL].astype(int)
        return df

    @staticmethod
    def _process_file_line(line: str) -> List[str]:
        """
        Processes the line from the data file
        :param line: line from the data file
        :return: a list containing the data from the row
        """
        row = MenuMatchPrecessor.EMPTY.join(line.split())
        row = row.replace(MenuMatchPrecessor.TAB, MenuMatchPrecessor.EMPTY)
        row = row.split(MenuMatchPrecessor.ORIG_DELIMINATOR)
        return row

    @staticmethod
    def _process_image_ingredient_mappings() -> Dict[str, List[str]]:
        """
        Processes the labels file and returns the data in a dictionary
        :return: a dictionary containing the image id mapped to a list of its ingredients
        """
        image_ingredient_mappings = {}
        with open(MenuMatchPrecessor.LABEL_FILE) as f:
            for line in f.readlines():
                img_data = MenuMatchPrecessor._process_file_line(line)
                image_ingredient_mappings[img_data[0]] = img_data[1:-1]
        return image_ingredient_mappings

    def _make_image_calorie_mapping_yml(self, item_info, image_ingredients_mapping):
        """
        Creates the image calorie mapping label file from the item info and image ingredients mapping
        :param item_info: information about the food in a dataframe
        :param image_ingredients_mapping: a dictionary containing the image id mapped to a list of its ingredients
        :return: None
        """
        image_calorie_mappings = {}
        for img_filename, labels in image_ingredients_mapping.items():
            total_calories = 0
            for label in labels:
                total_calories += item_info.loc[label][self.CALORIES_COL]
            img_id = os.path.split(img_filename)[0]
            image_calorie_mappings[img_id] = total_calories
        with open(self.dataset_path_creator.label_file, 'w') as file:
            yaml.dump(image_calorie_mappings, file)

    @staticmethod
    def _remove_missing_imgs(image_ingredient_mapping: Dict[str, List[str]]):
        """
        Removes the missing images from the image_ingredient_mapping
        :param image_ingredient_mapping: a dictionary containing the image id mapped to a list of its ingredients
        :return: None
        """
        for missing_img in MenuMatchPrecessor.MISSING_IMGS:
            if missing_img in image_ingredient_mapping:
                image_ingredient_mapping.pop(missing_img)  # this image is missing??
