from typing import Dict

import yaml
from tensorflow import Tensor

from cleaning.dataset import Dataset
from experiment.Food2Index import Food2Index
from scripts.preprocessing.processor import IMAGE_NAME_SEPARATOR


class FoodImagesDataset(Dataset):

    def __init__(self):
        """
        constructor
        """
        super().__init__('food_images', 'labels.yml')
        self._image_class_mappings = {}
        self.load_data()

    def get_label(self, image_name: str) -> Tensor:
        """
         get class corresponding to image
         :param image_name: name of the image
         :return: the food class
         """
        if IMAGE_NAME_SEPARATOR in image_name:
            image_name = image_name.split(IMAGE_NAME_SEPARATOR)[0]
        food_name = self.get_image_class_mappings()[image_name]
        return self.food2index.to_ingredients_tensor([food_name])

    def get_image_class_mappings(self) -> Dict[str, str]:
        """
        loads the image to class mappings from the label file
        :return: a dictionary of image, class pairs
        """
        return self._image_class_mappings
    
    def load_data(self):
        food2index = Food2Index()
        self._image_class_mappings = yaml.safe_load(open(self.label_file))
        for food_category in self._image_class_mappings.values():
            food2index.add(food_category)
        food2index.save()
