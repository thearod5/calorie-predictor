import os
from typing import Callable, List, Tuple

import tensorflow as tf
from tqdm import tqdm

from constants import BATCH_SIZE, N_EPOCHS, PROJECT_DIR
from src.experiment.cam.cam_dataset_converter import CamDatasetConverter
from src.experiment.cam.cam_logger import CamLogger
from src.experiment.cam.cam_loss import CamLoss
from src.experiment.cam.cam_loss_alpha import AlphaStrategy
from src.experiment.metric_provider import MetricProvider
from src.experiment.models.checkpoint_creator import get_checkpoint_path
from src.experiment.models.managers.model_manager import ModelManager


class CamTrainer:
    """
    Trains a model on dataset containing human mappings.
    """
    CAM_TRAINER_PATH = os.path.join(PROJECT_DIR, "results", "checkpoints", "CamTrainer")

    def __init__(self, model_manager: ModelManager, lr: float = 1e-4, weight_decay: float = 1e-6,
                 momentum: float = 0.9, checkpoint_name: str = None):
        """
        Initializes trainer with model and training parameters.
        :param model_manager: The manager of the model containing specification and builder.
        :param lr: The learning rate.
        :param weight_decay: The rate of decay of the optimizer.
        :param momentum: Optimizer momentum.s
        :param checkpoint_name: The name of the subdirectory inside of task checkpoints.
        """
        self.model_manager = model_manager
        self.model = model_manager.get_model()
        self.model_path = get_checkpoint_path(self.CAM_TRAINER_PATH, self.model_manager.get_model_name(),
                                              checkpoint_name=checkpoint_name)
        self.learning_rate = lr
        self.metrics: List[Tuple[str, Callable]] = [("mae", MetricProvider.mean_absolute_error),
                                                    ("avg_diff", MetricProvider.error_of_average)]
        self.feature_model = model_manager.get_feature_model()
        self.cam_logger = CamLogger(self.model_path, BATCH_SIZE)
        os.makedirs(self.model_path, exist_ok=True)
        print("Saving model to:", self.model_path)

    def save_model(self, prefix="") -> None:
        """
        Saves current model to model path.
        :param prefix: Prefix to print before logging statement.
        :return:None
        """
        self.model.save(self.model_path)
        message = " ".join([prefix, "Model Saved:", self.model_path])
        print(message)

    def train(self, training_data: CamDatasetConverter, validation_data: tf.data.Dataset, n_epochs: int = N_EPOCHS,
              alpha_strategy: AlphaStrategy = AlphaStrategy.ADAPTIVE) -> None:
        """
        Performs cam-training on data for some number of epochs.
        :param validation_data: The validation data to evaluate on.
        :param training_data: The data to train on containing calorie counts and human maps.
        :param n_epochs: The number of epochs to train for.
        :param alpha_strategy: The type of strategy to use for adjusting the alpha value.
        :return: None
        """

        cam_dataset = training_data.convert()
        cam_loss = CamLoss(self.feature_model, self.model_manager, n_epochs, alpha_strategy=alpha_strategy)

        for epoch in range(1, n_epochs + 1):
            self.perform_cam_epoch(cam_dataset, cam_loss, validation_data=validation_data)
            self.cam_logger.log_epoch(do_export=True)
            cam_loss.finish_epoch()

    def perform_cam_epoch(self, training_data, cam_loss: CamLoss,
                          validation_data: tf.data.Dataset = None,
                          n_evaluations: int = 4, eval_metric="mae",
                          metric_direction="lower") -> None:
        """
        Performs an epoch of cam training on data.
        :param training_data: The data to train on.
        :param cam_loss: The loss metric calculator.
        :param metric_direction: The direction in which the eval metric gets better.
        :param eval_metric: The metric to determine whether to save model.
        :param n_evaluations: The number of evaluations to perform during single epoch.
        :param validation_data: The data to evaluate on every n steps.
        :return: None
        """
        n_batches = len(training_data)
        epoch_evaluation = int(n_batches / n_evaluations)
        for batch_idx, (images, calories_expected, human_maps) in enumerate(tqdm(training_data)):
            y_true = (calories_expected, human_maps)
            self.perform_step(images, y_true, cam_loss)

            if (batch_idx + 1) % epoch_evaluation == 0 and validation_data:
                score = self.perform_evaluation(validation_data, use_tqdm=False)[eval_metric]
                score = round(score, 2)
                is_better = self.cam_logger.log_eval(score, metric_direction)
                if is_better:
                    message = "New Best Score:" + str(score)
                    self.save_model(prefix=message)

    def perform_step(self, x, y_true: Tuple[tf.Tensor, tf.Tensor], cam_loss: CamLoss):
        """
        Performs an optimizer step on the model given datum.
        :param x: The data to ask the model to predict.
        :param y_true: The true values of the data.
        :param cam_loss: The loss function manager.
        :return:
        """
        with tf.GradientTape() as tape:
            calories_predicted, feature_maps = self.feature_model(x, training=True)
            y_pred = (calories_predicted, feature_maps)
            calorie_loss, feature_loss, composite_loss = cam_loss.calculate_loss(y_pred, y_true)

        # 4. Compute and apply gradient
        grads = tape.gradient(composite_loss, self.feature_model.trainable_weights)
        cam_loss.optimizer.apply_gradients(zip(grads, self.feature_model.trainable_weights))

        calories_predicted_average = tf.math.reduce_mean(calories_predicted).numpy()
        self.cam_logger.log_step(composite_loss, calorie_loss, feature_loss, cam_loss.get_alpha(),
                                 predicted_average=calories_predicted_average)

    def perform_evaluation(self, test_data: tf.data.Dataset, use_tqdm: bool = True):
        """
        Evaluates current model on test dataset using initialized metrics.
        :param test_data: The dataset to evaluate on.
        :param use_tqdm: Whether to use the tqdm iterator which logs each iteration.
        :return: Dictionary of metric names to their values.
        """
        print("\nEvaluating...")
        calories_predicted = []
        calories_expected = []
        test_data_iterator = tqdm(test_data) if use_tqdm else test_data
        for batch_idx, (images, image_calories_expected) in enumerate(test_data_iterator):
            calories_predicted_local = self.model.predict(images)
            calories_predicted_local = tf.reshape(calories_predicted_local, (len(images)))
            image_calories_expected = tf.reshape(image_calories_expected, len(images))
            calories_predicted.extend(calories_predicted_local.numpy().tolist())
            calories_expected.extend(image_calories_expected.numpy().tolist())

        average_calories = round(sum(calories_expected) / len(calories_expected), 2)
        average_calories_predicted = round(sum(calories_predicted) / len(calories_predicted), 2)
        print("Average Calories:\t", average_calories, "Average Predicted:\t", average_calories_predicted, "\n")

        results = {}
        for metric_name, metric in self.metrics:
            metric_value = metric(calories_expected, calories_predicted)
            results[metric_name] = metric_value
        return results
