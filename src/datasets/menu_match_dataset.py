from typing import Dict

import yaml

from src.datasets.abstract_dataset import AbstractDataset
from src.datasets.dataset_path_creator import DatasetPathCreator


class MenuMatchDataset(AbstractDataset):
    DATASET_PATH_CREATOR = DatasetPathCreator(dataset_dir_name='menu_match', label_filename='total_calories.yml')

    def __init__(self):
        self._image_calorie_mappings = None
        super().__init__(self.DATASET_PATH_CREATOR)

    def get_image_calorie_mappings(self) -> Dict[str, float]:
        """
        loads the image to calorie mappings from the label file
        :return: a dictionary of image, calorie pairs
        """
        if self._image_calorie_mappings is None:
            self._image_calorie_mappings = yaml.safe_load(open(self.label_file))
        return self._image_calorie_mappings

    def get_label(self, image_name: str) -> float:
        """
        gets calories corresponding to image
        :param image_name: name of the image
        :return: the calories
        """
        return self.get_image_calorie_mappings()[image_name]
