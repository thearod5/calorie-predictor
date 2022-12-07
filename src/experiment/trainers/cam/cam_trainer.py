import os
from typing import Callable, List, Tuple

import tensorflow as tf
from tqdm import tqdm

from constants import BATCH_SIZE, N_EPOCHS, PROJECT_DIR
from src.experiment.metric_provider import MetricProvider
from src.experiment.models.managers.model_manager import ModelManager
from src.experiment.trainers.cam.cam_dataset_converter import CamDatasetConverter
from src.experiment.trainers.cam.cam_logger import CamLogger
from src.experiment.trainers.cam.cam_loss import CamLoss
from src.experiment.trainers.cam.cam_loss_alpha import AlphaStrategy


class CamTrainer:
    """
    Trains a model on dataset containing human mappings.
    """
    CAM_TRAINER_PATH = os.path.join(PROJECT_DIR, "results", "checkpoints", "CamTrainer")

    def __init__(self, model_manager: ModelManager, lr: float = 1e-4, weight_decay: float = 1e-6,
                 momentum: float = 0.9):
        """
        Initializes trainer with model and training parameters.
        :param model_manager: The manager of the model containing specification and builder.
        :param lr: The learning rate.
        :param weight_decay: The rate of decay of the optimizer.
        :param momentum: Optimizer momentum.s
        """
        self.model_manager = model_manager
        self.model = model_manager.get_model()
        self.learning_rate = lr
        self.metrics: List[Tuple[str, Callable]] = [("mae", MetricProvider.mean_absolute_error),
                                                    ("avg_diff", MetricProvider.error_of_average)]
        self.feature_model = model_manager.get_feature_model()
        self.cam_logger = CamLogger(self.model_manager.export_path, BATCH_SIZE)

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
        print("\n", "-" * 25)
        cam_dataset = training_data.convert()
        cam_loss = CamLoss(self.feature_model, self.model_manager, n_epochs, alpha_strategy=alpha_strategy)

        for epoch in range(1, n_epochs + 1):
            self.perform_cam_epoch(cam_dataset, cam_loss, validation_data=validation_data)
            self.cam_logger.log_epoch(do_export=True)
            cam_loss.finish_epoch()

    def perform_cam_epoch(self, training_data, cam_loss: CamLoss,
                          validation_data: tf.data.Dataset = None,
                          n_evaluations: int = 1, eval_metric="mae",
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
                score = self.perform_evaluation(validation_data)[eval_metric]
                score = round(score, 2)
                is_better = self.cam_logger.log_eval(score, metric_direction)
                if is_better:
                    message = "New Best Score:" + str(score)
                    self.model_manager.save_model()
                else:
                    message = "Score did not improve from %s (%s)" % (self.cam_logger.validation_score, score)
                    print("Previous best located at:", self.model_manager.export_path)
                print(message)

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

    def perform_evaluation(self, test_data: tf.data.Dataset):
        """
        Evaluates current model on test dataset using initialized metrics.
        :param test_data: The dataset to evaluate on.
        :return: Dictionary of metric names to their values.
        """
        print("\nEvaluating...")
        calories_predicted = self.model.predict(test_data)
        calories_predicted = tf.reshape(calories_predicted, (calories_predicted.shape[0])).numpy().tolist()
        calories_expected = [c_expected.numpy() for _, batch_y in test_data for c_expected in batch_y]

        results = {}
        for metric_name, metric in self.metrics:
            metric_value = metric(calories_expected, calories_predicted)
            results[metric_name] = round(metric_value, 2)

        print("Validation Metrics:", results, "\n")
        return results
