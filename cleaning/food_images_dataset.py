from typing import Dict

import yaml

from cleaning.dataset import Dataset
from scripts.preprocessing.processor import IMAGE_NAME_SEPARATOR


class FoodImagesDataset(Dataset):

    def __init__(self):
        """
        constructor
        """
        super().__init__('food_images', 'labels.yml')
        self._image_class_mappings = None

    def get_image_class_mappings(self) -> Dict[str, str]:
        """
        loads the image to class mappings from the label file
        :return: a dictionary of image, class pairs
        """
        if self._image_class_mappings is None:
            self._image_class_mappings = yaml.safe_load(open(self.label_file))
        return self._image_class_mappings

    def get_label(self, image_name: str) -> str:
        """
         get class corresponding to image
         :param image_name: name of the image
         :return: the food class
         """
        if IMAGE_NAME_SEPARATOR in image_name:
            image_name = image_name.split(IMAGE_NAME_SEPARATOR)[0]
        return self.get_image_class_mappings()[image_name]
