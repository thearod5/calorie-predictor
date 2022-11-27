from typing import Tuple

import tensorflow as tf

from src.experiment.cam.cam_dataset_converter import CamDatasetConverter
from src.experiment.models.managers.model_manager import ModelManager


class CamLoss:
    """
    Responsible for performing loss calculations for cam trainer.
    """

    def __init__(self, feature_model: tf.keras.Model, model_manager: ModelManager, total_epochs: int,
                 alpha_decay: float):
        self.feature_model = feature_model
        self.model_manager = model_manager
        self.criterion = tf.keras.losses.MeanSquaredError()
        self.criterion_hmap = tf.keras.losses.MeanSquaredError()
        self.alpha = 1
        self.alpha_decay = alpha_decay
        self.n_alpha_steps = 1 / alpha_decay
        self.n_alpha_updates = int(total_epochs / self.n_alpha_steps)
        self.n_epochs = 1
        self.optimizer = self.create_optimizer()

    @staticmethod
    def create_optimizer():
        """
        :return: tf.optimizers.Adam(learning_rate=0.01, epsilon=0.1)
        """
        return tf.optimizers.Adam(learning_rate=0.01, epsilon=0.1)

    def finish_epoch(self):
        """
        Records new epoch and updates alpha.
        :return: None
        """
        self.n_epochs += 1
        if self.n_epochs % self.n_alpha_updates == 0:
            self.alpha -= self.alpha_decay
            self.alpha = round(self.alpha, 2)
            self.optimizer = self.create_optimizer()

    @staticmethod
    def calculate_cam(feature_maps: tf.Tensor, feature_weights: tf.Tensor):
        """
        Calculates the feature maps relevant in model for each image in batch.
        :param feature_maps: The 2048 features maps extracted from model's last conv layer.
        :param feature_weights: The weight of each feature map to the activated class.
        :return: An feature map per image representing the weighted sum of the multiple feature maps
        per image.
        :rtype:
        """
        cams = []
        batch_size, feature_height, feature_width, n_features = feature_maps.shape
        for image_index in range(batch_size):
            cam = feature_maps[image_index]
            cam = tf.linalg.matmul(cam, feature_weights)
            cam = CamDatasetConverter.normalize(cam)
            cam = tf.reshape(cam, (feature_height, feature_width))
            cams.append(cam)
        return tf.stack(cams)

    def calculate_loss(self, y_pred: Tuple[tf.Tensor, tf.Tensor], y_true: Tuple[tf.Tensor, tf.Tensor]):
        """
        Calculates the loss between predicted and true labels.
        :param y_pred: Tuple of calorie counts and human maps for images predicted.
        :param y_true:Tuple of predicted calorie counts and model's feature maps.
        :return: Tuple representing calorie loss, cam loss, and composite loss.
        """
        calories_expected, hmaps = y_true
        calorie_predicted, feature_maps = y_pred
        calorie_predicted = tf.reshape(calorie_predicted, calories_expected.shape)
        calorie_loss = self.criterion(calories_expected, calorie_predicted)

        # 1. Compute weighted sum of the feature maps (Sk model) = model attention
        feature_weights = self.model_manager.get_feature_weights()
        batch_size, feature_height, feature_width, n_features = feature_maps.shape
        cams = CamLoss.calculate_cam(feature_maps, feature_weights)

        # 2. Get human maps = human attention
        hmaps = tf.reshape(hmaps, (batch_size, feature_height, feature_width))
        cam_loss = self.criterion_hmap(hmaps, cams)

        # 3. Calculate composite loss
        composite_loss = (self.alpha * calorie_loss) + ((1 - self.alpha) * cam_loss * calorie_loss)
        assert not tf.math.is_nan(composite_loss), "Composite loss is NAN."
        losses = calorie_loss, cam_loss, composite_loss
        losses = [tf.math.sqrt(loss) for loss in losses]
        return losses
