import os
from typing import Tuple

import tensorflow as tf
from PIL import Image
from keras_preprocessing.image import load_img

from constants import BATCH_SIZE
from src.datasets.abstract_dataset import AbstractDataset


class CamDatasetConverter:
    def __init__(self,
                 dataset: AbstractDataset,
                 map_dir_path: str,
                 map_size: Tuple[int, int] = (7, 7)):
        """
        Initializes with dataset and path to human maps.
        :param dataset: The dataset to convert.
        :param map_dir_path: Path to the human maps.
        :param map_size: The size to convert the maps to (defined by model features).
        """
        self.dataset = dataset
        self.map_dir_path = map_dir_path
        self.map_size = map_size

    def convert(self) -> tf.data.Dataset:
        """
        Converts AbstractDataset to HmapDataset.
        :return: Dataset containing images, calorie counts, and human annotated maps.
        """
        hmaps = []
        for image_name in self.dataset.get_image_names(with_extension=True):
            hmap = self.__read_hmap(image_name)
            hmap = self.pipeline(hmap)
            hmaps.append(hmap)
        hmap_dataset = tf.data.Dataset.from_tensor_slices(hmaps)
        images_dataset = self.dataset.get_images_dataset()
        labels_dataset = self.dataset.get_labels_dataset()
        return tf.data.Dataset.zip((images_dataset, labels_dataset, hmap_dataset)).batch(BATCH_SIZE)

    def __read_hmap(self, image_name: str) -> Image:
        """
        Reads hmap with name.
        :param image_name: The name of the hmap to read in.
        :return: Loaded image.
        """
        hmap_path = os.path.join(self.map_dir_path, image_name)
        return load_img(hmap_path)

    def pipeline(self, image):
        """
        Pre-processing pipeline for hmap images.
        :param image: The image to pre-process.
        :return: Resized, normalized, and grey scale image.
        """
        image = tf.image.resize(image, (self.map_size[0], self.map_size[1]))
        image = CamDatasetConverter.normalize(image)
        image = tf.convert_to_tensor(image)
        image = tf.image.rgb_to_grayscale(image)
        return image

    @staticmethod
    def normalize(tensor: tf.Tensor):
        """
        Performs min-max scaling on tensor so max value is 1 and min value is 0.
        :param tensor: The tensor to scale.
        :return: Scaled tensor.
        """
        tensor_max = tf.reduce_max(tensor)
        tensor_min = tf.reduce_min(tensor)
        tensor_range = tf.subtract(tensor_max, tensor_min)
        if tensor_range == 0:
            return tensor
        tensor_without_min = tf.subtract(
            tensor,
            tensor_min
        )
        return tf.divide(tensor_without_min, tensor_range)
