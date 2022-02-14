from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy
from tensorflow.python.data import AUTOTUNE

from constants import *
import tensorflow as tf
import os


class Dataset:

    def __init__(self, dataset_dirname: str, label_filename: str):
        self.dataset_dir = os.path.join(DATA_DIR, dataset_dirname)
        self.label_file = os.path.join(self.dataset_dir, label_filename)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self._training_data = None
        self._testing_data = None
        self._image_filenames = None
        self._images = None

    @abstractmethod
    def get_label(self, image_name: str) -> any:
        pass

    def get_image_filenames(self) -> tf.data.Dataset:
        if self._image_filenames is None:
            self._image_filenames = [os.path.join(self.image_dir, filename) for filename in os.listdir(self.image_dir)]
        return self._image_filenames

    def get_labels(self) -> tf.data.Dataset:
        labels = []
        for name in self.get_image_filenames():
            name = name.split(os.sep)[-1].split(EXT_SEP)[0]  # get only the image name
            labels.append(self.get_label(name))
        return tf.data.Dataset.from_tensor_slices(tf.ragged.constant(labels))  # TODO convert to dataset

    def decode_img(self, img: object) -> object:
        img = tf.io.decode_jpeg(img, channels=3)
        return tf.image.resize(img, IMAGE_SIZE)

    def process_path(self, file_path: tf.data.Dataset) -> object:
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img

    def get_images(self) -> tf.data.Dataset:
        if self._images is None:
            path_ds = tf.data.Dataset.from_tensor_slices(self.get_image_filenames())
            self._images = path_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)
        return self._images

    def get_train_and_val_datasets(self, val_size=0) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        dataset = tf.data.Dataset.zip((self.get_images(), self.get_labels()))
        image_count = len(self.get_image_filenames())
        val_size = int(image_count * val_size)
        ds = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds.take(image_count - val_size), ds.skip(val_size)
