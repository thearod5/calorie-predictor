import os
from abc import abstractmethod
from typing import Dict, List, Tuple, Optional

from tensorflow.python.keras.preprocessing.image import save_img

from datasets.dataset import Dataset

ImageLabel = Dict[str, str]
ProcessingPath = Tuple[str, str]  # Represents the input path to an image and the output path after resizing
ProcessingPaths = List[ProcessingPath]


class ProcessingSettings:
    def __init__(self, show_errors, throw_errors):
        self.show_errors = show_errors
        self.throw_errors = throw_errors


class AbstractProcessor:
    BAR = "-" * 50

    def __init__(self, image_dir: str):
        self.dataset = Dataset("", "")
        self.dataset.image_dir = image_dir
        self.n_processed = 0
        self.n_failed = 0

    @abstractmethod
    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        pass

    def pre_process(self):
        pass

    def post_process(self):
        pass

    def print_status(self, override=True):
        end = "\r" if override else "\n"
        print("Processed: ", self.n_processed, "Failed: ", self.n_failed, end=end)

    def print_bar(self):
        print(self.BAR)

    def process(self, settings: ProcessingSettings) -> None:
        self.print_bar()
        print("Processing : " + self.__class__.__name__)
        self.pre_process()
        image_paths = self.dataset.get_image_paths()
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
        self.print_status(override=False)
        self.post_process()
        return True

    @staticmethod
    def read_resize_save_image(input_image_path: str, output_image_path: str, settings: ProcessingSettings):
        try:
            file_name = os.path.split(output_image_path)[1]
            if os.path.isfile(output_image_path) or file_name[0] == ".":
                return
            if not os.path.isfile(input_image_path):
                raise Exception("No image found at path: ", input_image_path)
            image = Dataset.decode_image_from_path(input_image_path)
            save_img(output_image_path, image)
        except Exception as e:
            if settings.show_errors:
                print("FAILED:", input_image_path)
                print(e)

                if settings.throw_errors:
                    raise e
            return e

    @staticmethod
    def create_generic_single_output(entry_name: str, output_folder: str) -> ProcessingPaths:
        image_file_name = os.path.split(entry_name)[1]
        output_image_path = os.path.join(output_folder, image_file_name)
        return [(entry_name, output_image_path)]
