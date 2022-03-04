import yaml

from cleaning.dataset import Dataset
import csv
from enum import IntEnum, Enum
from typing import *


class Dish:

    def __init__(self, name, calories, mass, ingredients=None):
        self.name = name
        self.mass = mass
        self.calories = calories
        self.ingredients = ingredients


class Mode(Enum):
    CALORIE = 'calories'
    MASS = 'mass'
    INGREDIENTS = 'ingredients'


class NutritionDataset(Dataset):
    id_index = 0
    calorie_index = 1
    mass_index = 2
    num_features = 6

    def __init__(self, dataset_num: int = 1, mode: Mode = Mode.MASS):
        """
        constructor
        :param dataset_num: number of the dataset (either 1 or 2)
        :param mode: mode to determine which labels will be used
        """
        label_file = 'dish_metadata_cafe' + str(dataset_num) + '.csv'
        super().__init__('nutrition5k', label_file)
        self._dishes = None
        self._ingredients = None
        self._mode = mode

    def get_label(self, image_name: str) -> Union[float, List[str]]:
        """
        gets label (determined by the mode) corresponding to image
        :param image_name: name of the image
        :return: the label(s)
        """
        mappings = self.get_dishes() if image_name in self.get_dishes() else self.get_ingredients()
        dish = mappings[image_name]
        return getattr(dish, self._mode.value)

    def set_mode(self, mode: Mode) -> None:
        """
        sets the mode for which labels will be used
        :param mode: the mode (i.e. calories, mass...)
        :return: None
        """
        self._mode = mode

    def get_dishes(self) -> Dict[str, Dish]:
        """
        gets the dictionary mapping dish id to the dish
        :return: dict with dish id, Dish pairs
        """
        if self._dishes is None:
            self.load_data()
        return self._dishes

    def get_ingredients(self) -> Dict[str, Dish]:
        """
        gets the dictionary mapping ingredient id to the ingredient
        :return: dict with id, Dish pairs
        """
        if self._ingredients is None:
            self.load_data()
        return self._ingredients

    def load_data(self) -> None:
        """
        loads the data from the label file
        :return: None
        """
        self._dishes = {}
        self._ingredients = {}
        with open(self.label_file, newline='') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                dish_ingr = []
                num_ingredients = int((len(row) / self.num_features))
                for i in range(1, num_ingredients):
                    index = i * self.num_features
                    ingr_id = row[index + self.id_index]
                    self._ingredients[ingr_id] = Dish(ingr_id, float(row[index + self.calorie_index]),
                                                      float(row[index + self.mass_index]))
                    dish_ingr.append(ingr_id)
                dish_id = row[self.id_index]
                dish = Dish(dish_id, float(row[self.calorie_index]), float(row[self.mass_index]), dish_ingr)
                self._dishes[dish_id] = dish
