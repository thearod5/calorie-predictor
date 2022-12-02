from src.datasets.unimib_dataset import UnimibDataset
from src.preprocessing.base_processor import BaseProcessor, ProcessingPaths, ProcessingSettings


class UnimibProcessor(BaseProcessor):

    def __init__(self):
        super().__init__(UnimibDataset.DATASET_PATH_CREATOR, "UNIMIB2016-images")

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        """
        Creates the path to the output for each image
        :param entry_name: the name of the image
        :return: the paths
        """
        return self.create_generic_single_output(entry_name, self.dataset_path_creator.image_dir)

    def post_process(self, settings: ProcessingSettings):
        self._copy_label_file()
