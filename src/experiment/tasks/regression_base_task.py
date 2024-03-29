from abc import ABC
from typing import Any, List, Tuple

import numpy as np
import tensorflow as tf
from keras.losses import mean_absolute_error

from constants import N_EPOCHS
from src.experiment.models.managers.model_manager import ModelManager
from src.experiment.tasks.base_task import AbstractTask, logger
from src.experiment.tasks.task_type import TaskType
from src.logging_utils.utils import format_name_val_info


class RegressionBaseTask(AbstractTask, ABC):
    task_type = TaskType.REGRESSION
    loss_function = "mse"
    metric = "mae"
    task_mode = "min"

    def __init__(self, model_manager: ModelManager, n_outputs=1, n_epochs=N_EPOCHS):
        """
        Represents a Regression Tasks
        :param model_manager: the model to use for the task
        :param n_outputs: the number of nodes for the output layer
        :param n_epochs: the number of epochs to run training for
        """
        super().__init__(model_manager, n_outputs, n_epochs)

    def eval(self, _=None):
        """
        Evaluates the task based on the predictions
        :param _: not needed (required to override super)
        :return: None
        """
        y_true, y_pred = self.get_predictions(self.get_test_data())
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten()).numpy()
        logger.info(format_name_val_info("Test Mean Absolute Error", mae))

    def get_predictions(self, data) -> Tuple[List, Any]:
        """
        Gets the models predictions
        :param data: data to predict on
        :return: the expected results and those predicted
        """
        y_true, y_pred = super().get_predictions(data)
        y_true = list(map(lambda v: v.numpy(), y_true))  # unpacks 1D vector into single number
        y_true = np.array(y_true)
        return y_true, y_pred

    def create_model(self) -> tf.keras.Model:
        return self.model_manager.create_model((self.task_type, self.task_name), n_outputs=self.n_outputs,
                                               pre_trained_model=self.is_pretrained)
