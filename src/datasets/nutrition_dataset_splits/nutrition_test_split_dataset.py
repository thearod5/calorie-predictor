import os

from src.datasets.dataset_path_creator import DatasetPathCreator
from src.datasets.nutrition_dataset import NutritionDataset


class NutritionTestSplitDataset(NutritionDataset):
    DATASET_DIR_NAME = os.path.join(NutritionDataset.DATASET_PATH_CREATOR.dataset_dir, "test")
    DATASET_PATH_CREATOR = DatasetPathCreator(dataset_dir_name=DATASET_DIR_NAME, label_filename='')
