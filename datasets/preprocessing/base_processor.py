import os
from abc import abstractmethod
from typing import Dict, List, Tuple, Optional
from collections import NamedTuple
from tensorflow.python.keras.preprocessing.image import save_img

from datasets.abstract_dataset import AbstractDataset, DatasetPathCreator

ImageLabel = Dict[str, str]
ProcessingPath = Tuple[str, str]  # Represents the input path to an image and the output path after resizing
ProcessingPaths = List[ProcessingPath]


class ProcessingSettings:
    def __init__(self, show_errors: bool, throw_errors: bool):
        """
        Contains the settings for the processor
        :param show_errors: if True, prints errors
        :param throw_errors: if True, errors are not caught and will stop the program
        """
        self.show_errors = show_errors
        self.throw_errors = throw_errors


class BaseProcessor:
    BAR = "-" * 50

    def __init__(self, dataset_path_creator: DatasetPathCreator, source_image_dir: str):
        """
        Handles processing of the datasets
        :param dataset_path_creator: handles making the paths to the input data for a dataset
        :param source_image_dir: the directory containing the source (unprocessed) images
        """
        self.dataset_path_creator = dataset_path_creator
        self.image_dir = os.path.join(dataset_path_creator.source_dir, source_image_dir)
        self.n_processed = 0
        self.n_failed = 0

    @abstractmethod
    def create_output_paths(self, path_to_input_images: str) -> ProcessingPaths:
        """
        Creates the output paths for each input image dir
        :param path_to_input_images: path to input images
        :return: a list of tuple containing the input and output image path pairs
        """
        pass

    def pre_process(self, settings: ProcessingSettings):
        """
        Can be overridden by child class to perform pre-processing
        :param settings: contains the appropriate settings for the processing run
        :return: None
        """
        pass

    def post_process(self, settings: ProcessingSettings):
        """
        Can be overridden by child class to perform steps after pre-processing is run
        :param settings: contains the appropriate settings for the processing run
        :return: None
        """
        pass

    def print_status(self, override: bool = True):
        """
        Prints the status of the pre-processing
        :param override: If True, uses \r as the newline character else \n is used
        :return: None
        """
        end = "\r" if override else "\n"
        print("Processed: ", self.n_processed, "Failed: ", self.n_failed, end=end)

    def print_bar(self):
        """
        Prints a divider bar
        :return: None
        """
        print(self.BAR)

    def process(self, settings: ProcessingSettings) -> Optional[Exception]:
        """
        Performs processing on the dataset
        :param settings: contains the appropriate settings for the processing run
        :return: the exception if one occurred
        """
        self.print_bar()
        print("Processing : " + self.__class__.__name__)
        try:
            self.pre_process(settings)
            self.resize_images(AbstractDataset.get_image_paths(self.image_dir), settings)
            self.post_process(settings)
            self.print_status(override=False)
        except Exception as e:
            print("Processing %s has failed" % self.__class__.__name__)
            print(e)
            return e

    def resize_images(self, image_paths: List, settings: ProcessingSettings):
        """
        Resizes all images in a dataset
        :param image_paths: list of image paths
        :param settings: contains the appropriate settings for the processing run
        :return: None
        """
        for entry_name in image_paths:
            entry_outputs = self.create_output_paths(entry_name)
            for entry_output in entry_outputs:
                input_path, output_path = entry_output
                error = self.read_resize_save_image(input_path, output_path, settings)
                if error is not None:
                    self.n_failed += 1
                self.n_processed += 1
            if self.n_processed % 5000 == 0:
                self.print_status()

    @staticmethod
    def read_resize_save_image(input_image_path: str, output_image_path: str, settings: ProcessingSettings) -> Optional[Exception]:
        """
        Reads, resizes and saves image
        :param input_image_path: the path to the input image
        :param output_image_path: path of where to save the image
        :param settings: contains the appropriate settings for the processing run
        :return: the exception if one occurred
        """
        try:
            file_name = os.path.split(output_image_path)[1]
            if os.path.isfile(output_image_path) or file_name[0] == ".":
                return
            if not os.path.isfile(input_image_path):
                raise Exception("No image found at path: ", input_image_path)
            image = AbstractDataset.decode_image_from_path(input_image_path)
            save_img(output_image_path, image)
        except Exception as e:
            if settings.show_errors:
                print("FAILED:", input_image_path)
                print(e)

                if settings.throw_errors:
                    raise e
            return e

    @staticmethod
    def create_generic_single_output(path_to_input_images: str, output_folder: str) -> ProcessingPaths:
        """
        Creates a tuple containing the input and output image path for a dataset with a single path per image
        :param path_to_input_images: path to input images
        :param output_folder: the output dir name
        :return: a tuple containing the input and output image path inside of a list (in case of multiple paths)
        """
        image_file_name = os.path.split(path_to_input_images)[1]
        output_image_path = os.path.join(output_folder, image_file_name)
        return [(path_to_input_images, output_image_path)]
