import yaml

from cleaning.dataset import Dataset
import csv
from enum import IntEnum
from typing import *


class Dish:

    def __init__(self, name, calories, mass, ingredients=None):
        self.name = name
        self.mass = mass
        self.calories = calories
        self.ingredients = ingredients


class Mode(IntEnum):
    CALORIE = 0
    MASS = 1
    INGREDIENTS = 2


class Nutrition5kDataset(Dataset):
    id_index = 0
    calorie_index = 1
    mass_index = 2
    num_features = 6

    def __init__(self, dataset_num=1):
        label_file = 'dish_metadata_cafe' + str(dataset_num) + '.csv'
        super().__init__('nutrition5k', label_file)
        self._dishes = None
        self._image_calorie_mappings = None
        self._image_mass_mappings = None
        self._ingredients = None
        self._mode = Mode.MASS

    def get_label(self, image_name: str) -> Union[float, List[str]]:
        mappings = self.get_dishes() if image_name in self.get_dishes() else self.get_ingredients()
        dish = mappings[image_name]
        if self._mode == Mode.MASS:
            label = dish.mass
        elif self._mode == Mode.CALORIE:
            label = dish.calories
        else:
            label = dish.ingredients
        return label

    def set_mode(self, mode: Mode) -> None:
        self._mode = mode

    def get_dishes(self) -> Dict[str, Dish]:
        if self._dishes is None:
            self.load_data()
        return self._dishes

    def get_ingredients(self) -> Dict[str, Dish]:
        if self._ingredients is None:
            self.load_data()
        return self._ingredients

    def load_data(self) -> None:
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
