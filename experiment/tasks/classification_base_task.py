from abc import ABC

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from constants import N_EPOCHS
from experiment.Food2Index import Food2Index
from experiment.models.model_manager import ModelManager
from experiment.tasks.task_type import TaskType
from logging_utils.utils import format_eval_results, format_header


class ClassificationBaseTask(AbstractTask, ABC):
    task_type = TaskType.CLASSIFICATION
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    metric = "accuracy"
    task_mode = "max"

    def __init__(self, model_manager: ModelManager, log_path: str, n_outputs=1, n_epochs=N_EPOCHS, load_weights=True,
                 load_on_init=True):
        """
        Represents a Classification Task
        :param model_manager: the model to use for the task
        :param n_outputs: the number of nodes for the output layer
        :param n_epochs: the number of epochs to run training for
        :param load_weights: if True, loads existing weights
        :param load_on_init: if True, loads the model in task __init__
        """
        super().__init__(model_manager, log_path, n_outputs, n_epochs, load_weights, load_on_init)

    def eval(self, dataset_name: str = None):
        """
         Evaluates the task based on the predictions
         :param dataset_name: the name of the dataset to use for evaluation
         :return: None
         """
        logger.info(format_header("Eval"))
        data = self.get_test_data() if dataset_name is None else self.get_eval_dataset(dataset_name)
        food2index = Food2Index()

        y_test, y_pred = self.get_predictions(data)  # no validation data on any class. task

        predictions = []
        labels = []

        class_tp = {}
        class_fp = {}
        class_fn = {}

        for test_vector, pred_vector in zip(y_test, y_pred):
            pred = np.argmax(pred_vector)
            label = np.argmax(test_vector)

            pred_name = food2index.get_ingredient(pred)
            label_name = food2index.get_ingredient(label)

            predictions.append(pred_name)
            labels.append(label_name)

            if pred == label:
                self.increment_dict_entry(class_tp, label_name)
            else:
                self.increment_dict_entry(class_fp, pred_name, label_name)
                self.increment_dict_entry(class_fn, label_name, pred_name)

        self.print_metrics(labels, predictions)
        logger.info(format_eval_results(class_tp, "TP"))
        logger.info(format_eval_results(class_fp, "FP"))
        logger.info(format_eval_results(class_fn, "FN"))

    @staticmethod
    def print_metrics(labels, predictions):
        """
        Preints the
        :param labels:
        :param predictions:
        :return:
        """
        matrix = confusion_matrix(labels, predictions)
        FP = matrix.sum(axis=0) - np.diag(matrix)
        FN = matrix.sum(axis=1) - np.diag(matrix)
        TP = np.diag(matrix)
        TN = matrix.sum() - (FP + FN + TP)
        logger.info("Predictions: %s" % len(predictions))
        logger.info("False Positive: %s" % FP.sum())
        logger.info("False Negatives: %s" % FN.sum())
        logger.info("True Positive: %s" % TP.sum())
        logger.info("True Negative: %s" % TN.sum())
