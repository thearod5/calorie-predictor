from enum import Enum

import tensorflow as tf

from src.experiment.models.managers.ensemble_model_manager import EnsembleModelManager
from src.experiment.models.managers.resnet_model_manager import ResNetModelManager
from src.experiment.models.managers.test_model_manager import TestModelManager
from src.experiment.models.managers.vgg_model_manager import VGGModelManager
from src.experiment.models.managers.xception_model_manager import XceptionModelManager


class ModelManagers(Enum):
    VGG = VGGModelManager
    RESNET = ResNetModelManager
    XCEPTION = XceptionModelManager
    TEST = TestModelManager
    ENSEMBLE = EnsembleModelManager

    @staticmethod
    def create_from_model_path(model_path: str):
        model = tf.keras.models.load_model(model_path)
        print(model)


PRE_TRAINED_MODELS = [ModelManagers.VGG, ModelManagers.RESNET, ModelManagers.XCEPTION]
