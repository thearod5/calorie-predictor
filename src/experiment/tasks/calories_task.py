import os

import tensorflow as tf

from constants import N_EPOCHS, TEST_SPLIT_SIZE, get_cam_path
from src.datasets.menu_match_dataset import MenuMatchDataset
from src.datasets.nutrition_dataset import Mode, NutritionDataset
from src.experiment.cam.cam_dataset_converter import CamDatasetConverter
from src.experiment.cam.cam_trainer import CamTrainer
from src.experiment.models.managers.model_manager import ModelManager
from src.experiment.tasks.regression_base_task import RegressionBaseTask


class CaloriePredictionTask(RegressionBaseTask):

    def __init__(self, model_manager: ModelManager, n_epochs=N_EPOCHS, use_cam=False, **cam_args):
        """
        Represents a Calorie Prediction Task
        :param model_manager: the model to use for the task
        :param n_epochs: the number of epochs to run training for
        """
        super().__init__(model_manager, n_epochs=n_epochs, load_weights=False)
        menu_match_dataset = MenuMatchDataset()
        nutrition_dataset = NutritionDataset(Mode.CALORIE)
        train, validation = menu_match_dataset.split_to_train_test(TEST_SPLIT_SIZE)
        test = nutrition_dataset.split_to_train_test().pop()

        self.cam_path = os.path.join(get_cam_path(), menu_match_dataset.dataset_path_creator.name)
        self.use_cam = use_cam
        self.dataset = menu_match_dataset
        self._train = train
        self._validation = validation
        self._test = test
        self.trainer = CamTrainer(model_manager, **cam_args)

    def train(self):
        if self.use_cam:
            cam_dataset_converter = CamDatasetConverter(self.dataset, self.cam_path)
            self.trainer.train(cam_dataset_converter, self._validation)
        else:
            super().train()

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

    def get_eval_dataset(self, name: str) -> tf.data.Dataset:
        """
        Gets the dataset to use for evaluation
        :param name: the name of the dataset
        :return: the dataset
        """
        raise NotImplementedError("Don't know different between test or eval datasets - alberto.")
