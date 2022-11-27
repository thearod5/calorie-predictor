import os

from constants import IMAGE_DIR, PATH_TO_PROJECT, get_data_dir


class DatasetPathCreator:

    def __init__(self, dataset_dir_name: str, label_filename: str):
        """
        Handles constructing the paths to the data
        :param dataset_dir_name: name of the directory for the dataset
        :param label_filename: name of the file containing all image labels
        """
        self.name = dataset_dir_name
        self.dataset_dir = self._create_dataset_dir_path(dataset_dir_name)
        self.label_file = self._create_label_file_path(self.dataset_dir, label_filename)
        self.image_dir = self._create_image_dir_path(self.dataset_dir)
        self.source_dir = self._create_source_dataset_path(dataset_dir_name)

    @staticmethod
    def _create_source_dataset_path(dataset_dir_name: str) -> str:
        """
        Create the path to the unprocessed data
        :param dataset_dir_name: name of the base directory for the dataset
        :return: the path
        """
        return os.path.join(PATH_TO_PROJECT, dataset_dir_name)

    @staticmethod
    def _create_dataset_dir_path(dataset_dir_name: str) -> str:
        """
        Creates the base path to the dataset directory
        :param dataset_dir_name: name of the directory for the dataset
        :return: the path
        """
        data_dir = get_data_dir()
        return os.path.join(PATH_TO_PROJECT, data_dir, dataset_dir_name)

    @staticmethod
    def _create_label_file_path(dataset_dir: str, label_filename: str) -> str:
        """
        Creates the path to the label file
        :param dataset_dir: name of the directory for the dataset
        :param label_filename: name of the file containing all image labels
        :return: the path
        """
        return os.path.join(dataset_dir, label_filename)

    @staticmethod
    def _create_image_dir_path(dataset_dir: str) -> str:
        """
        Creates the path to the image directory
        :param dataset_dir: name of the directory for the dataset
        :return: the path
        """
        return os.path.join(dataset_dir, IMAGE_DIR)
