from typing import Dict

import yaml

from cleaning.dataset import Dataset


class MenuMatchDataset(Dataset):

    def __init__(self):
        """
        constructor
        """
        super().__init__('menu_match', 'total_calories.yml')
        self._image_calorie_mappings = None

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


