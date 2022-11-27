import tensorflow as tf

from constants import N_EPOCHS, TEST_SPLIT_SIZE
from src.datasets.eucstfd_dataset import EucstfdDataset
from src.datasets.nutrition_dataset import Mode, NutritionDataset
from src.experiment.models.managers.model_manager import ModelManager
from src.experiment.tasks.regression_base_task import RegressionBaseTask


class MassPredictionTask(RegressionBaseTask):

    def __init__(self, model_manager: ModelManager, n_epochs=N_EPOCHS):
        super().__init__(model_manager, n_epochs=n_epochs)
        self._train, self._validation = EucstfdDataset().split_to_train_test(TEST_SPLIT_SIZE)
        self._test = NutritionDataset(Mode.MASS).split_to_train_test().pop()

    def get_training_data(self) -> tf.data.Dataset:
        """
        :return: The dataset to use for training.
        """
        return self._train

    def get_validation_data(self) -> tf.data.Dataset:
        """
        :return: The dataset used for validation.
        """
        return self._validation

    def get_test_data(self) -> tf.data.Dataset:
        """
        :return: The dataset used for testing.
        """
        return self._test

    def get_eval_dataset(self, name: str) -> [str]:
        """
        Gets the dataset to use for evaluation
        :param name: the name of the dataset
        :return: the dataset
        """
        pass  # TODO
