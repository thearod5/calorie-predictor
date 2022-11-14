import os
from typing import Tuple

import tensorflow as tf
from PIL import Image
from keras_preprocessing.image import load_img

from datasets.abstract_dataset import AbstractDataset


class CamDatasetConverter:
    def __init__(self,
                 dataset: AbstractDataset,
                 map_dir_path: str,
                 map_size: Tuple[int, int] = (7, 7)):
        self.dataset = dataset
        self.map_dir_path = map_dir_path
        self.map_size = map_size
        self.pipeline = self.__create_pipeline()

    def convert(self) -> tf.data.Dataset:
        hmaps = []
        for image_name in self.dataset.get_image_names(with_extension=True):
            hmap = self.__read_hmap(image_name)
            hmap = self.pipeline(hmap)
            # hmap = tf.image.rgb_to_grayscale(hmap)
            hmaps.append(hmap)
        hmap_dataset = tf.data.Dataset.from_tensor_slices(hmaps)
        images_dataset = self.dataset.get_images_dataset()
        labels_dataset = self.dataset.get_labels_dataset()
        return tf.data.Dataset.zip((images_dataset, labels_dataset, hmap_dataset)).batch(3)

    def __read_hmap(self, image_name: str) -> Image:
        hmap_path = os.path.join(self.map_dir_path, image_name)
        return load_img(hmap_path)

    def __create_pipeline(self):
        def pipeline(image):
            image = tf.image.resize(image, (self.map_size[0], self.map_size[1]))
            image = CamDatasetConverter.normalize(image)
            image = tf.convert_to_tensor(image)
            return image

        return pipeline

    @staticmethod
    def normalize(tensor):
        return tf.divide(
            tf.subtract(
                tensor,
                tf.reduce_min(tensor)
            ),
            tf.subtract(
                tf.reduce_max(tensor),
                tf.reduce_min(tensor)
            )
        )
