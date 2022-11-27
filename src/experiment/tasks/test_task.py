import tensorflow as tf

from constants import N_EPOCHS, TEST_SPLIT_SIZE
from src.datasets.menu_match_dataset import MenuMatchDataset
from src.experiment.models.managers.model_manager import ModelManager
from src.experiment.tasks.regression_base_task import RegressionBaseTask


class TestTask(RegressionBaseTask):

    def __init__(self, model_manager: ModelManager, log_path: str, n_epochs=N_EPOCHS):
        super().__init__(model_manager, log_path, n_epochs=n_epochs)
        dataset = MenuMatchDataset()
        train, validation = dataset.split_to_train_test(TEST_SPLIT_SIZE)
        self._train = train
        self._validation = validation

    def get_training_data(self):
        return self._train

    def get_validation_data(self):
        return self._validation

    def get_test_data(self):
        # NO TEST DATA
        return self._validation

    def get_eval_dataset(self, name: str) -> tf.data.Dataset:
        # NO EVAL DATA
        return self._validation
