import os
from abc import abstractmethod
from typing import *

import cv2 as cv
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE

from constants import *


def decode_image_from_path(file_path: str) -> object:
    """
    reads in an image file and process it
    :param file_path: the image file path
    :return: processed image
    """
    if isinstance(file_path, str) and ".h264" in file_path:
        input_video = cv.VideoCapture(file_path)
        ret, frame = input_video.read()
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = tf.convert_to_tensor(img_rgb)
    else:
        img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img, channels=N_CHANNELS)
    return tf.image.resize(img, IMAGE_SIZE)


class Dataset:

    def __init__(self, dataset_dirname: str, label_filename: str):
        """
        constructor
        :param dataset_dirname: name of the directory for the dataset
        :param label_filename: name of the file containing all image labels
        """
        self.dataset_dir = os.path.join(DATA_DIR, dataset_dirname)
        self.label_file = os.path.join(self.dataset_dir, label_filename)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self._image_paths = None
        self._images = None

    @abstractmethod
    def get_label(self, image_name: str) -> any:
        """
        gets label corresponding to image
        :param image_name: name of the image
        :return: the label(s)
        """
        pass

    def get_image_paths(self) -> List:
        """
        gets a list of all image filenames
        :return: a list of the image filenames
        """
        if self._image_paths is None:
            self._image_paths = [os.path.join(self.image_dir, filename) for filename in
                                 os.listdir(self.image_dir) if filename[0] != "."]  # ignore system files
        return self._image_paths

    def get_labels(self) -> tf.data.Dataset:
        """
        makes a dataset of all image labels
        :return: a tensor flow dataset of all labels
        """
        labels = []
        for name in self.get_image_paths():
            name = name.split(os.sep)[-1].split(EXT_SEP)[0]  # get only the image name
            labels.append(self.get_label(name))
        return tf.data.Dataset.from_tensor_slices(tf.ragged.constant(labels))

    def get_images(self) -> tf.data.Dataset:
        """
        makes a dataset of all processed images
        :return: a dataset of all images
        """
        if self._images is None:
            path_ds = tf.data.Dataset.from_tensor_slices(self.get_image_paths())
            self._images = path_ds.map(decode_image_from_path, num_parallel_calls=AUTOTUNE)
        return self._images

    def split_to_train_test(self, test_split_size: float = 0) -> List[tf.data.Dataset]:
        """
        gets a zipped dataset of image, label pairs which are split into two (if spit_size = 0, only one dataset is created)
        :param test_split_size: the percent of the data to go in one split
        :return: list of dataset splits
        """
        image_count = len(self.get_image_paths())
        ds = tf.data.Dataset.zip((self.get_images(), self.get_labels()))
        ds = ds.shuffle(buffer_size=image_count, seed=RANDOM_SEED)
        d_splits = self.split_dataset(ds, image_count, test_split_size) if test_split_size > 0 else [ds]
        return [self.prepare_dataset(d_split) for d_split in d_splits]

    def prepare_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        applies batching and prefetching to dataset
        :param dataset: original dataset
        :return: the prepared dataset
        """
        ds = dataset.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def split_dataset(self, dataset: tf.data.Dataset, image_count: int, test_split_size: float) -> Tuple[
        tf.data.Dataset, tf.data.Dataset]:
        """
        splits the dataset into two
        :param dataset: original dataset
        :param image_count: the number of images in the dataset
        :param test_split_size: the percent of the data to go in one split
        :return: the two dataset splits
        """
        test_split_size = int(image_count * test_split_size)
        return dataset.take(image_count - test_split_size), dataset.skip(test_split_size)
