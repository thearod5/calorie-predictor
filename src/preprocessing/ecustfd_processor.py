from src.datasets.eucstfd_dataset import EucstfdDataset
from src.preprocessing.base_processor import BaseProcessor, ProcessingPaths, ProcessingSettings


class EcustfdProcessor(BaseProcessor):

    def __init__(self):
        """
        Handles processing for the eucstfd dataset
        """
        super().__init__(EucstfdDataset.DATASET_PATH_CREATOR, "JPEGImages")

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        """
        Creates the path to the output for each image
        :param entry_name: the name of the image
        :return: the paths
        """
        return self.create_generic_single_output(entry_name, self.dataset_path_creator.image_dir)

    def post_process(self, settings: ProcessingSettings):
        self._copy_label_file()
