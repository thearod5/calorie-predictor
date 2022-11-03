from typing import Dict

import yaml

from datasets.abstract_dataset import AbstractDataset, DatasetPathCreator


class MenuMatchDataset(AbstractDataset):

    dataset_paths_creator = DatasetPathCreator(dataset_dirname='menu_match', label_filename='total_calories.yml')

    def __init__(self):
        self._image_calorie_mappings = None
        super().__init__(self.dataset_paths_creator)

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


