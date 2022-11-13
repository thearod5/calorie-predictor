from datasets.menu_match_dataset import MenuMatchDataset
from datasets.nutrition_dataset import Mode, NutritionDataset
from constants import N_EPOCHS, TEST_SPLIT_SIZE
from experiment.tasks.regression_base_task import RegressionBaseTask

import tensorflow as tf


class CaloriePredictionTask(RegressionBaseTask):

    def __init__(self, base_model, n_epochs=N_EPOCHS):
        """
        Represents a Calorie Prediction Task
        :param base_model: the model to use for the task
        :param n_epochs: the number of epochs to run training for
        """
        super().__init__(base_model, n_epochs=n_epochs)
        dataset = NutritionDataset(Mode.CALORIE)
        train, validation = dataset.split_to_train_test(TEST_SPLIT_SIZE)
        test_dataset = MenuMatchDataset()
        self._train = train
        self._validation = validation
        self._test = test_dataset.split_to_train_test().pop()

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

    def get_eval_dataset(self, name: str) -> [str]:
        """
        Gets the dataset to use for evaluation
        :param name: the name of the dataset
        :return: the dataset
        """
        pass  # TODO
