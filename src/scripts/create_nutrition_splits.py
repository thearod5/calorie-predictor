import os
import shutil

from constants import PROJECT_DIR
from src.datasets.nutrition_dataset import NutritionDataset
from src.datasets.nutrition_dataset_splits.nutrition_test_split_dataset import NutritionTestSplitDataset
from src.datasets.nutrition_dataset_splits.nutrition_train_split_dataset import NutritionTrainSplitDataset


def copy_files_to_split_dir(split_file_path: str, destination_image_dir: str):
    original_image_dir = NutritionDataset.DATASET_PATH_CREATOR.image_dir
    train_split_content = read_file(split_file_path)
    for dish_id in train_split_content.splitlines():
        image_path = os.path.join(original_image_dir, dish_id + "-A.jpg")

        if os.path.isfile(image_path):
            shutil.copy(image_path, destination_image_dir)
        else:
            print("File not found: %s" % image_path)


def read_file(file_path: str):
    with open(file_path, "r") as file:
        return file.read()


if __name__ == "__main__":
    split_dir_path = os.path.join(PROJECT_DIR, "data", "nutrition5k", "splits")
    train_split_path = os.path.join(split_dir_path, "train_ids.txt")
    test_split_path = os.path.join(split_dir_path, "test_ids.txt")

    copy_files_to_split_dir(train_split_path, NutritionTrainSplitDataset.DATASET_PATH_CREATOR.image_dir)
    copy_files_to_split_dir(test_split_path, NutritionTestSplitDataset.DATASET_PATH_CREATOR.image_dir)
