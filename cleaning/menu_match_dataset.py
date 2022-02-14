from typing import Dict

import yaml

from cleaning.dataset import Dataset


class MenuMatchDataset(Dataset):

    def __init__(self):
        super().__init__('menu_match', 'total_calories.yml')

        self._image_ingredient_mappings = None
        self._image_calorie_mappings = None
        self.food_info = None
        self.calorie_labels = None

    def get_image_calorie_mappings(self) -> Dict[str, float]:
        if self._image_calorie_mappings is None:
            self._image_calorie_mappings = yaml.safe_load(open(self.label_file))
        return self._image_calorie_mappings

    def get_label(self, image_name: str) -> float:
        return self.get_image_calorie_mappings()[image_name]


