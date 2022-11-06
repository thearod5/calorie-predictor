import os
from typing import Dict, List

import yaml
from tensorflow import Tensor

from datasets.abstract_dataset import AbstractDataset, DatasetPathCreator
from experiment.Food2Index import Food2Index
from constants import IMAGE_NAME_SEPARATOR


class FoodImagesDataset(AbstractDataset):
    dataset_paths_creator = DatasetPathCreator(dataset_dirname='food_images', label_filename='labels.yml')

    def __init__(self):
        self._image_class_mappings = {}
        super().__init__(self.dataset_paths_creator)

    @staticmethod
    def get_image_paths(image_dir: str) -> List:
        """
        Returns images by listing directories of categories and listing the images
        within those directories.
        :return: a list of the image filenames
        """
        paths = []
        for dir_name in os.listdir(image_dir):
            if dir_name[0] == ".":
                continue
            path_to_dir = os.path.join(image_dir, dir_name)
            for file_name in os.listdir(path_to_dir):
                if file_name[0] == ".":
                    continue
                path_to_file = os.path.join(path_to_dir, file_name)
                paths.append(path_to_file)
        return paths

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

    def load_data(self) -> None:
        """
        Loads the data for the dataset
        :return: None
        """
        food2index = Food2Index()
        self._image_class_mappings = yaml.safe_load(open(self.label_file))
        for food_category in self._image_class_mappings.values():
            food2index.add(food_category)
        food2index.save()
