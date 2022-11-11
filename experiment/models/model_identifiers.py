from enum import Enum

import tensorflow as tf

from experiment.models.ensemble_model import EnsembleModel
from experiment.models.test_model import TestModel


class BaseModel(Enum):
    VGG = tf.keras.applications.VGG19
    RESNET = tf.keras.applications.ResNet50
    XCEPTION = tf.keras.applications.Xception
    TEST = TestModel
    ENSEMBLE = EnsembleModel


PRE_TRAINED_MODELS = [BaseModel.VGG.value, BaseModel.RESNET.value, BaseModel.XCEPTION.value]
