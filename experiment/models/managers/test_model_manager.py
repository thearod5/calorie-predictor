"""
A models for basic testing of inputs and outputs.
"""
from keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow.python.layers.base import Layer

from constants import INPUT_SHAPE
from experiment.models.managers.model_manager import ModelManager

N_FEATURES = 32
KERNEL_SIZE = (3, 3)


class TestModelManager(ModelManager):
    def __init__(self):
        super().__init__()

    def get_model_constructor(self):
        model = Sequential()
        model.add(layers.Conv2D(N_FEATURES, KERNEL_SIZE, activation="relu", input_shape=INPUT_SHAPE))
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        return lambda: model

    @staticmethod
    def _get_feature_layer(model: Model) -> Layer:
        return model.layers[1].conv

    def get_parameters(self):
        return self.get_model().parameters()
