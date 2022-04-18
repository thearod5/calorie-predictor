import csv
import os
from enum import Enum
from typing import *

from tensorflow import Tensor

from cleaning.dataset import Dataset, get_name_from_path
from constants import LOG_SKIPPED_ENTRIES
from scripts.preprocessing.processor import IMAGE_NAME_SEPARATOR
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


class Dish:

    def __init__(self, name, calories, mass, ingredients):
        self.name = name
        self.mass = mass
        self.calories = calories
        self.ingredients = ingredients

    def str(self) -> str:
        return "%s: C{%s} M{%s} I{%s}" % (self.name, self.calories, self.mass, self.ingredients)


class Mode(Enum):
    CALORIE = 'calories'
    MASS = 'mass'
    INGREDIENTS = 'ingredients'


DatasetMap = {
    1: [Mode.CALORIE, Mode.MASS],
    2: [Mode.INGREDIENTS]
}


def is_mode_value_valid(mode_value: Union[float, list]):
    if isinstance(mode_value, float) and mode_value < 1:
        return False
    if isinstance(mode_value, list) and len(mode_value) == 0:
        return False
    return True


class NutritionDataset(Dataset):
    id_index = 0
    calorie_index = 1
    mass_index = 2
    num_features = 6

    def __init__(self, mode: Mode):
        """
        constructor
        :param dataset_num: number of the dataset (either 1 or 2)
        :param mode: mode to determine which labels will be used
        """
        dataset_dirname = 'nutrition5k'
        super().__init__(dataset_dirname, "")
        self._dishes = {}
        self._mode = mode
        self._label_files = ["dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"]
        self._load_data()

    def get_image_paths(self):
        """
        Overrides parent paths to filter out images whose corresponding dishes do not
        have sufficient or valid data.
        :return:
        """
        return list(filter(lambda p: self._get_image_dish(get_name_from_path(p)) is not None,
                           super().get_image_paths()))

    def get_label(self, image_name: str) -> Union[float, Tensor]:
        """
        gets label (determined by the mode) corresponding to image
        :param image_name: name of the image
        :return: the label(s)
        """
        dish = self._get_image_dish(image_name)
        if dish is None:
            raise Exception("No labels found for image:" + image_name)
        label = getattr(dish, self._mode.value)
        if self._mode == Mode.INGREDIENTS:
            return self.food2index.to_ingredients_tensor(label)
        return label

    def _get_image_dish(self, image_name: str) -> Union[Dish, None]:
        if IMAGE_NAME_SEPARATOR in image_name:
            image_name = image_name.split(IMAGE_NAME_SEPARATOR)[0]  # removes the camera identifier
        if image_name not in self._dishes:
            return None
        return self._dishes[image_name]

    def _load_data(self) -> None:
        """
        loads the data from the label file
        :return: None
        """
        self._dishes = {}
        processed_ids = []
        for label_file_name in self._label_files:
            path_to_label_file = os.path.join(self.dataset_dir, label_file_name)
            with open(path_to_label_file, newline='') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    dish_id = row[self.id_index]
                    dish = self._parse_row_into_dish(row)
                    dish_mode_value = getattr(dish, self._mode.value)
                    if dish_id in processed_ids or not is_mode_value_valid(dish_mode_value):
                        if LOG_SKIPPED_ENTRIES:
                            logging.debug(dish_id, ": was already processed or has invalid value for mode", self._mode)
                        continue
                    self._dishes[dish_id] = self._parse_row_into_dish(row)
                    processed_ids.append(dish_id)
        self.food2index.save()

    def _parse_row_into_dish(self, row) -> Dish:

        # 1. Get calories and mass
        dish_id = row[self.id_index]
        dish_calories = float(row[self.calorie_index])
        dish_mass = float(row[self.mass_index])

        # 2. Get ingredients
        n_ingredients = len(row) / self.num_features
        dish_ingredients = []
        for ingredient_index in range(1, int(n_ingredients)):
            ingredient_name_index = ingredient_index * NutritionDataset.num_features
            ingredient_name = row[ingredient_name_index]
            self.food2index.add(ingredient_name)
            dish_ingredients.append(ingredient_name)

        # 3. Create dish and save
        return Dish(dish_id, dish_calories, dish_mass, dish_ingredients)
