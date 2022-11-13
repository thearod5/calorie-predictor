import tensorflow as tf

from constants import N_EPOCHS, TEST_SPLIT_SIZE
from datasets.menu_match_dataset import MenuMatchDataset
from datasets.nutrition_dataset import Mode, NutritionDataset
from experiment.cam.cam_trainer import CamTrainer
from experiment.models.managers.model_manager import ModelManager
from experiment.tasks.regression_base_task import RegressionBaseTask


class CaloriePredictionTask(RegressionBaseTask):

    def __init__(self, model_manager: ModelManager, n_epochs=N_EPOCHS, use_cam=False):
        """
        Represents a Calorie Prediction Task
        :param model_manager: the model to use for the task
        :param n_epochs: the number of epochs to run training for
        """
        super().__init__(model_manager, n_epochs=n_epochs, load_weights=False)
        dataset = MenuMatchDataset()
        train, validation = dataset.split_to_train_test(TEST_SPLIT_SIZE)
        test_dataset = NutritionDataset(Mode.CALORIE)
        self.use_cam = use_cam
        self.dataset = dataset
        self._train = train
        self._validation = validation
        self._test = test_dataset.split_to_train_test().pop()
        self.trainer = CamTrainer(model_manager)

    def train(self):
        if self.use_cam:
            self.trainer.train(self.dataset)
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

    def get_eval_dataset(self, name: str) -> [str]:
        """
        Gets the dataset to use for evaluation
        :param name: the name of the dataset
        :return: the dataset
        """
        pass  # TODO
