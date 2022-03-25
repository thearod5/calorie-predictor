import json
import os

import numpy as np

from constants import PROJECT_DIR


class IngredientIndexMap:
    def __init__(self):
        self.path_to_dict = os.path.join(PROJECT_DIR, "data", "ingredients", "index.json")
        self.index = {}
        if not os.path.isfile(self.path_to_dict):
            self.save()
        else:
            with open(self.path_to_dict) as json_file:
                self.index = json.load(json_file)
        self.next_index = len(self.index.values())

    def get_index(self, ingredient_name: str) -> int:
        return self.index[ingredient_name]

    def get_ingredient(self, index: int) -> str:
        for ingredient_name, ingredient_index in self.index.items():
            if index == ingredient_index:
                return ingredient_name

    def add(self, ingredient_name: str):
        if ingredient_name not in self.index:
            self.index[ingredient_name] = self.next_index
            self.next_index += 1

    def save(self):
        with open(self.path_to_dict, "w") as outfile:
            json.dump(self.index, outfile)

    def ingredients2vector(self, ingredients: [str]):
        vector = np.zeros(shape=(self.__len__()))
        for ingredient in ingredients:
            vector[self.get_index(ingredient)] = 1
        return vector

    def __len__(self):
        return len(self.index)
