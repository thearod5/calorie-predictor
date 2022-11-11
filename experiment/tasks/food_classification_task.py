from typing import Callable, Dict, List

import numpy as np
import tensorflow as tf

from constants import CLASSIFICATION_DATASETS, N_EPOCHS, TEST_SPLIT_SIZE
from datasets.abstract_dataset import AbstractDataset
from datasets.food_images_dataset import FoodImagesDataset
from datasets.nutrition_dataset import Mode, NutritionDataset
from datasets.unimib_dataset import UnimibDataset
from experiment.Food2Index import Food2Index
from experiment.models.model_identifiers import BaseModel
from experiment.tasks.classification_base_task import ClassificationBaseTask


class FoodClassificationTask(ClassificationBaseTask):
    dataset_constructors: Dict[str, Callable[[], AbstractDataset]] = {
        "unimib": lambda: UnimibDataset(),
        "food_images": lambda: FoodImagesDataset(),
        "nutrition5k": lambda: NutritionDataset(mode=Mode.INGREDIENTS)
    }

    def __init__(self, base_model: BaseModel, n_epochs=N_EPOCHS, training_datasets: [str] = CLASSIFICATION_DATASETS):
        """
         Represents the Food Classification Task
         :param base_model: the model to use for the task
         :param n_epochs: the number of epochs to run training for
         :param training_datasets: datasets to use for training
         """
        super().__init__(base_model, n_outputs=len(Food2Index()), n_epochs=n_epochs)
        datasets = self.get_datasets(training_datasets)
        dataset, image_count = self.combine_datasets(datasets)
        d_splits = list(
            map(AbstractDataset.prepare_dataset, AbstractDataset.split_dataset(dataset, image_count, TEST_SPLIT_SIZE)))
        train, validation = d_splits[0], d_splits[1]
        self._train = train
        self._validation = validation
        self._test = None

    def get_eval_dataset(self, name: str) -> tf.data.Dataset:
        """
        Gets the dataset to use for evaluation
        :param name: the name of the dataset
        :return: the dataset
        """
        if name not in self.dataset_constructors:
            raise Exception("%s is not one of %s" % (name, ", ".join(self.dataset_constructors.keys())))
        return AbstractDataset.prepare_dataset(self.dataset_constructors[name]().get_dataset(shuffle=False))

    def get_datasets(self, dataset_names: List[str]) -> [AbstractDataset]:
        """
                Gets the dataset objects from a list of their names
                :param dataset_names: a list of the names of the datasets to get
                :return: a list of datasets
                """
        return [self.dataset_constructors[d_name]() for d_name in self.dataset_constructors if d_name in dataset_names]

    def get_training_data(self) -> tf.data.Dataset:
        """
        Gets the dataset to use for training
        :return: the dataset
        """
        return self._train

    def get_validation_data(self) -> tf.data.Dataset:
        """
        Gets the dataset to use for validation
        :return: the dataset
        """
        return self._validation

    def get_test_data(self) -> tf.data.Dataset:
        """
        Gets the dataset to use for testing
        :return: the dataset
        """
        return self._test

    def get_total_food_count(self) -> int:
        """
        Gets the total food count across all datasets
        :return: the total number of foods in the datasets
        """
        count = 0
        for dataset in [self._train, self._test, self._validation]:
            if dataset is None:
                continue
            count += len(self.get_food_counts(dataset))
        return count

    @staticmethod
    def get_food_counts(dataset: tf.data.Dataset) -> Dict[str, int]:
        """
        Creates a dictionary of foods mapped to the number of pictures of that food in the dataset
        :param dataset: the dataset to get food counts for
        :return:  dictionary mapping food name to the number of pictures of that food in the dataset
        """
        food2index = Food2Index()
        food2count = {}
        for batch_images, batch_labels in dataset:
            for batch_index in range(batch_images.shape[0]):
                label = batch_labels[batch_index]
                label_indices = np.where(label == 1)[0]
                for label_index in label_indices:
                    food_name = food2index.get_ingredient(label_index)
                    if food_name in food2count:
                        food2count[food_name] += 1
                    else:
                        food2count[food_name] = 1

        return food2count
