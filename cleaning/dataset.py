from abc import abstractmethod
from typing import *
from tensorflow.python.data import AUTOTUNE

from constants import *
import tensorflow as tf
import os


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
        self._image_filenames = None
        self._images = None

    @abstractmethod
    def get_label(self, image_name: str) -> any:
        """
        gets label corresponding to image
        :param image_name: name of the image
        :return: the label(s)
        """
        pass

    def get_image_filenames(self) -> List:
        """
        gets a list of all image filenames
        :return: a list of the image filenames
        """
        if self._image_filenames is None:
            self._image_filenames = [os.path.join(self.image_dir, filename) for filename in os.listdir(self.image_dir)]
        return self._image_filenames

    def get_labels(self) -> tf.data.Dataset:
        """
        makes a dataset of all image labels
        :return: a tensor flow dataset of all labels
        """
        labels = []
        for name in self.get_image_filenames():
            name = name.split(os.sep)[-1].split(EXT_SEP)[0]  # get only the image name
            labels.append(self.get_label(name))
        return tf.data.Dataset.from_tensor_slices(tf.ragged.constant(labels))

    def decode_img(self, img: object) -> object:
        """
        decodes and resizes image
        :param img: the image
        :return: the decoded image
        """
        img = tf.io.decode_jpeg(img, channels=3)
        return tf.image.resize(img, IMAGE_SIZE)

    def process_path(self, file_path: str) -> object:
        """
        reads in an image file and process it
        :param file_path: the image filepath
        :return: processed image
        """
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img

    def get_images(self) -> tf.data.Dataset:
        """
        makes a dataset of all processed images
        :return: a dataset of all images
        """
        if self._images is None:
            path_ds = tf.data.Dataset.from_tensor_slices(self.get_image_filenames())
            self._images = path_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)
        return self._images

    def get_datasets_splits(self, split_size: float = 0) -> List[tf.data.Dataset]:
        """
        gets a zipped dataset of image, label pairs which are split into two (if spit_size = 0, only one dataset is created)
        :param split_size: the percent of the data to go in one split
        :return: list of dataset splits
        """
        image_count = len(self.get_image_filenames())
        ds = tf.data.Dataset.zip((self.get_images(), self.get_labels()))
        ds = ds.shuffle(buffer_size=image_count, seed=RANDOM_SEED)
        datasets = self.split_dataset(ds, image_count, split_size) if split_size > 0 else [ds]
        return [self.prepare_dataset(dataset) for dataset in datasets]

    def prepare_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        applies batching and prefetching to dataset
        :param dataset: original dataset
        :return: the prepared dataset
        """
        ds = dataset.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def split_dataset(self, dataset: tf.data.Dataset, image_count: int, split_size: float) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        splits the dataset into two
        :param dataset: original dataset
        :param image_count: the number of images in the dataset
        :param split_size: the percent of the data to go in one split
        :return: the two dataset splits
        """
        split_size = int(image_count * split_size)
        return dataset.take(image_count - split_size), dataset.skip(split_size)
