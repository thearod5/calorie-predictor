import os

import tensorflow as tf
import torch
from PIL import Image
from tensorflow.keras.layers import Reshape

from datasets.abstract_dataset import AbstractDataset


class CamDatasetConverter:
    def __init__(self,
                 dataset: AbstractDataset,
                 map_dir_path: str,
                 map_size: int = 7):
        self.dataset = dataset
        self.map_dir_path = map_dir_path
        self.map_size = map_size

    def convert(self) -> tf.data.Dataset:
        dataset_cams = []
        for image_name in self.dataset.get_image_names(with_extension=True):
            human_map = self.__read_human_map(image_name)
            cam_map = self.__convert_map_to_cam(human_map)
            dataset_cams.append(cam_map)
        return tf.data.Dataset.zip((self.dataset.get_images(), self.dataset.get_labels(), dataset_cams))

    def __read_human_map(self, image_name: str) -> Image:
        map_path = os.path.join(self.map_dir_path, image_name)
        with Image.open(map_path) as im:
            return im

    def __convert_map_to_cam(self, human_map: Image):
        transform_human_map = self.__map_transform()(human_map)
        transform_human_map = torch.squeeze(transform_human_map)
        transform_human_map = transform_human_map - torch.min(transform_human_map)
        return transform_human_map / torch.max(transform_human_map)

    def __map_transform(self):
        transforms = torch.nn.Sequential(
            Reshape([self.map_size, self.map_size])
            # TODO: add conversion to tensor
            # TODO: add conversion to float
        )
        return torch.jit.script(transforms)
