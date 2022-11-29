import json
import os

import yaml

from constants import PROJECT_DIR
from src.datasets.food_images_dataset import FoodImagesDataset
from src.experiment.Food2Index import Food2Index

if __name__ == "__main__":
    mode = "create"

    if mode == "create":
        food_images = FoodImagesDataset()

        with open(food_images.label_file, 'r') as file:
            food_images_labels = yaml.safe_load(file)
            food_set = set()
            for food in food_images_labels.values():
                food_set.add(food)

            food_set = sorted(list(food_set))
            food2index = {food_index: food for food_index, food in enumerate(food_set)}
            print(food2index)

            label_file = os.path.join(PROJECT_DIR, "data", "ingredients", "index.json")
            with open(label_file, "w") as label_output_file:
                json.dump(food2index, label_output_file)
    elif mode == "play":
        food_index = Food2Index()
        print(len(food_index))
