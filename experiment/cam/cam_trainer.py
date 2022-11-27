import os
from typing import Callable, List, Tuple

import tensorflow as tf
from tqdm import tqdm

from constants import BATCH_SIZE, PROJECT_DIR
from experiment.cam.cam_dataset_converter import CamDatasetConverter
from experiment.cam.cam_loss import CamLoss
from experiment.cam.cam_state import CamState
from experiment.metric_provider import MetricProvider
from experiment.models.managers.model_manager import ModelManager


class CamTrainer:
    """
    Trains a model on dataset containing human mappings.
    """
    CAM_TRAINER_PATH = os.path.join(PROJECT_DIR, "results", "checkpoints", "CamTrainer")

    def __init__(self, model_manager: ModelManager, lr: float = 1e-4, weight_decay: float = 1e-6,
                 momentum: float = 0.9, load_model=True, model_path: str = None):
        """
        Initializes trainer with model and training parameters.
        :param model_manager: The manager of the model containing specification and builder.
        :param lr: The learning rate.
        :param weight_decay: The rate of decay of the optimizer.
        :param momentum: Optimizer momentum.s
        """
        self.model_manager = model_manager
        self.model = model_manager.get_model()
        self.model_path = os.path.join(self.CAM_TRAINER_PATH, self.model_manager.get_model_name())
        self.learning_rate = lr
        self.metrics: List[Tuple[str, Callable]] = [("mae", MetricProvider.mean_absolute_error),
                                                    ("avg_diff", MetricProvider.error_of_average)]
        self.feature_model = model_manager.create_feature_model()
        self.cam_state = CamState(self.model_path, BATCH_SIZE)

        assert os.path.exists(self.model_path), "Model path does not exists:" + self.model_path

        self.save_or_load_model(load_model)

    def save_or_load_model(self, load_model: bool) -> None:
        """
        Loads model from path or saves initialized model to path.
        :param load_model: Whether to load model if it exists.
        :type load_model:
        :return: None
        """
        if not self.model_exists():
            self.save_model()
        elif load_model:
            print("loading from model:", self.model_path)
            self.model = tf.keras.models.load_model(self.model_path)

    def model_exists(self) -> bool:
        """
        Returns whether current model exists in model path.
        :return: Boolean representing existence of model.
        """
        check_path = os.path.join(self.model_path, "keras_metadata.pb")
        return os.path.exists(check_path)

    def save_model(self, prefix="") -> None:
        """
        Saves current model to model path.
        :param prefix: Prefix to print before logging statement.
        :return:None
        """
        self.model.save(self.model_path)
        message = " ".join([prefix, "Model Saved:", self.model_path])
        print(message)

    def train(self, training_data: CamDatasetConverter, validation_data: tf.data.Dataset, n_epochs: int = 30,
              alpha_decay: float = .2) -> None:
        """
        Performs cam-training on data for some number of epochs.
        :param validation_data: The validation data to evaluate on.
        :param training_data: The data to train on containing calorie counts and human maps.
        :param n_epochs: The number of epochs to train for.
        :param alpha_decay: The amount to decrease the alpha by incrementally across epochs.
        :return: None
        """

        cam_dataset = training_data.convert()
        cam_loss = CamLoss(self.feature_model, self.model_manager, n_epochs, alpha_decay)

        for epoch in range(1, n_epochs + 1):
            self.perform_cam_epoch(cam_dataset, cam_loss, validation_data=validation_data)
            self.cam_state.log_epoch(do_export=True)
            cam_loss.finish_epoch()

    def perform_cam_epoch(self, training_data, cam_loss: CamLoss,
                          validation_data: tf.data.Dataset = None,
                          evaluate_steps: int = 100, eval_metric="mae",
                          metric_direction="lower") -> None:
        """
        Performs an epoch of cam training on data.
        :param training_data: The data to train on.
        :param cam_loss: The loss metric calculator.
        :param metric_direction: The direction in which the eval metric gets better.
        :param eval_metric: The metric to determine whether to save model.
        :param evaluate_steps: The number of training steps to perform before evaluation.
        :param validation_data: The data to evaluate on every n steps.
        :return: None
        """

        for batch_idx, (images, calories_expected, human_maps) in enumerate(tqdm(training_data)):
            y_true = (calories_expected, human_maps)
            self.perform_step(images, y_true, cam_loss)

            if batch_idx > 0 and batch_idx % evaluate_steps == 0 and validation_data:
                score = self.evaluate(validation_data)[eval_metric]
                is_better = self.cam_state.log_eval(score, metric_direction)
                if is_better:
                    print("New Best Score:", score)
                    self.save_model(prefix="New best!")

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
            calorie_loss, cam_loss, composite_loss = cam_loss.calculate_loss(y_pred, y_true)

        # 4. Compute and apply gradient
        grads = tape.gradient(composite_loss, self.feature_model.trainable_weights)
        cam_loss.optimizer.apply_gradients(zip(grads, self.feature_model.trainable_weights))

        calories_predicted_average = tf.math.reduce_mean(calories_predicted).numpy()
        self.cam_state.log_step(composite_loss, calorie_loss, cam_loss, predicted_average=calories_predicted_average)

    def evaluate(self, test_data: tf.data.Dataset):
        """
        Evaluates current model on test dataset using initialized metrics.
        :param test_data: The dataset to evaluate on.
        :return: Dictionary of metric names to their values.
        """
        calories_predicted = []
        calories_expected = []
        for batch_idx, (images, image_calories_expected) in enumerate(tqdm(test_data)):
            calories_predicted_local = self.model.predict(images)
            calories_predicted_local = tf.reshape(calories_predicted_local, (len(images)))
            image_calories_expected = tf.reshape(image_calories_expected, len(images))
            calories_predicted.extend(calories_predicted_local.numpy().tolist())
            calories_expected.extend(image_calories_expected.numpy().tolist())

        print("\naverage calories:", sum(calories_expected) / len(calories_expected))
        print("average predicted:", sum(calories_predicted) / len(calories_predicted))
        results = {}
        for metric_name, metric in self.metrics:
            metric_value = metric(calories_expected, calories_predicted)
            results[metric_name] = metric_value
        return results
