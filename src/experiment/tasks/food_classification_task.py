from typing import Dict

import numpy as np
import tensorflow as tf

from constants import CLASSIFICATION_DATASETS, N_EPOCHS, TEST_SPLIT_SIZE
from src.datasets.food_images_dataset import FoodImagesDataset
from src.experiment.Food2Index import Food2Index
from src.experiment.models.managers.model_manager import ModelManager
from src.experiment.tasks.classification_base_task import ClassificationBaseTask


class FoodClassificationTask(ClassificationBaseTask):

    def __init__(self, model_manager: ModelManager, n_epochs=N_EPOCHS,
                 training_datasets: [str] = CLASSIFICATION_DATASETS):
        """
         Represents the Food Classification Task
         :param model_manager: the model to use for the task
         :param n_epochs: the number of epochs to run training for
         :param training_datasets: datasets to use for training
         """
        super().__init__(model_manager, n_outputs=len(Food2Index()), n_epochs=n_epochs)
        self._train, self._validation = FoodImagesDataset().split_to_train_test(TEST_SPLIT_SIZE)
        self._test = None

    def get_eval_dataset(self, name: str) -> tf.data.Dataset:
        """
        Gets the dataset to use for evaluation
        :param name: the name of the dataset
        :return: the dataset
        """
        raise ValueError("Evaluation is not defined for food classification due to conflicting classes.")

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
