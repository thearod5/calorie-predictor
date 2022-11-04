from abc import abstractmethod
from typing import *

import cv2 as cv
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE

from constants import *
from experiment.Food2Index import Food2Index


class DatasetPathCreator:

    def __init__(self, dataset_dirname: str, label_filename: str):
        """
        Handles constructing the paths to the data
        :param dataset_dirname: name of the directory for the dataset
        :param label_filename: name of the file containing all image labels
        :param data_dir: name of directory containing the data
        """
        self.dataset_dir = self._create_dataset_dir_path(dataset_dirname)
        self.label_file = self._create_label_file_path(self.dataset_dir, label_filename)
        self.image_dir = self._create_image_dir_path(self.dataset_dir)
        self.source_dir = self._create_source_dataset_path

    @staticmethod
    def _create_source_dataset_path(dataset_dirname: str) -> str:
        """
        Create the path to the unprocessed data
        :param dataset_dirname: name of the base directory for the dataset
        :return: the path
        """
        return os.path.join(PATH_TO_PROJECT, dataset_dirname)

    @staticmethod
    def _create_dataset_dir_path(dataset_dirname: str) -> str:
        """
        Creates the base path to the dataset directory
        :param dataset_dirname: name of the directory for the dataset
        :return: the path
        """
        data_dir = get_data_dir()
        return os.path.join(PATH_TO_PROJECT, data_dir, dataset_dirname)

    @staticmethod
    def _create_label_file_path(dataset_dir: str, label_filename: str) -> str:
        """
        Creates the path to the label file
        :param dataset_dirname: name of the directory for the dataset
        :param label_filename: name of the file containing all image labels
        :return: the path
        """
        return os.path.join(dataset_dir, label_filename)

    @staticmethod
    def _create_image_dir_path(dataset_dir: str) -> str:
        """
        Creates the path to the image directory
        :param dataset_dirname: name of the directory for the dataset
        :return: the path
        """
        return os.path.join(dataset_dir, IMAGE_DIR)


class AbstractDataset:
    def __init__(self, dataset_path_creator: DatasetPathCreator):
        """
        Represents a dataset for calorie predictor
        :param dataset_path_creator: handles making the paths for a given dataset
        """
        self.dataset_dir = dataset_paths_creator.dataset_dir
        self.label_file = dataset_paths_creator.label_file
        self.image_dir = dataset_paths_creator.image_dir
        self.image_paths = get_image_paths(self.image_dir)
        self._images = None
        self.food2index = Food2Index()
        self.load_data()

    def load_data(self) -> None:
        """
        Performs the necessary logic to load the data for the dataset
        :return: None
        """
        pass

    @abstractmethod
    def get_label(self, image_name: str) -> any:
        """
        gets label corresponding to image
        :param image_name: name of the image
        :return: the label(s)
        """
        pass

    @staticmethod
    def get_image_paths(image_dir: str) -> List:
        """
        gets a list of all image filenames
        :param image_dir: path to the image directory
        :return: a list of the image filenames
        """
        return [os.path.join(self.image_dir, filename) for filename in
                os.listdir(self.image_dir) if
                filename[0] != "." and filename != ""]  # ignore system files

    def get_image_names(self) -> List[str]:
        """
        Gets the image names
        :return: a list of image names
        """
        return list(map(self.get_name_from_path, self.image_paths))

    def get_labels(self) -> tf.data.Dataset:
        """
        makes a dataset of all image labels
        :return: a tensor flow dataset of all labels
        """
        labels = []
        image_names = self.get_image_names()
        for name in image_names:
            labels.append(self.get_label(name))
        if not isinstance(labels[0], tf.Tensor):
            labels = tf.ragged.constant(labels)
        return tf.data.Dataset.from_tensor_slices(labels)

    def get_images(self) -> tf.data.Dataset:
        """
        makes a dataset of all processed images
        :return: a dataset of all images
        """
        if self._images is None:
            path_ds = tf.data.Dataset.from_tensor_slices(self.image_paths)
            self._images = path_ds.map(self.decode_image_from_path, num_parallel_calls=AUTOTUNE)
        return self._images

    def split_to_train_test(self, test_split_size: float = 0, shuffle=True) -> List[tf.data.Dataset]:
        """
        gets a zipped dataset of image, label pairs which are split into two (if spit_size = 0, only one dataset is created)
        :param shuffle: shuffles data if True
        :param test_split_size: the percent of the data to go in one split
        :return: list of dataset splits
        """
        image_count = len(self.image_paths)
        ds = self.get_dataset(shuffle)
        d_splits = self.split_dataset(ds, image_count, test_split_size) if test_split_size > 0 else [ds]
        return [self.prepare_dataset(d_split) for d_split in d_splits]

    def get_dataset(self, shuffle=True) -> tf.data.Dataset:
        """
        gets a zipped dataset of image, label pairs
       :param shuffle: shuffles data if True
       :return: the dataset
       """
        image_count = len(self.image_paths)
        ds = tf.data.Dataset.zip((self.get_images(), self.get_labels()))
        if shuffle:
            ds = ds.shuffle(buffer_size=image_count, seed=RANDOM_SEED)
        return ds

    @staticmethod
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

    @staticmethod
    def get_name_from_path(path: str) -> str:
        """
        Extracts the name from the path
        :param path: the path
        :return: the name
        """
        return os.path.split(entry_name)[-1].split(EXT_SEP)[0]

    @staticmethod
    def split_dataset(dataset: tf.data.Dataset, image_count: int, test_split_size: float) -> Tuple[
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

    @staticmethod
    def prepare_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        applies batching and prefetching to dataset
        :param dataset: original dataset
        :return: the prepared dataset
        """
        ds = dataset.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
