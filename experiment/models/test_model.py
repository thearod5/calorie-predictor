"""
A models for basic testing of inputs and outputs.
"""
from tensorflow.keras import layers, models
import tensorflow as tf
from constants import INPUT_SHAPE

N_FEATURES = 32
KERNEL_SIZE = (3, 3)


class TestModel(models.Sequential):

    def __init__(self):
        uper().__init__()
        self.add(layers.Conv2D(N_FEATURES, KERNEL_SIZE, activation="relu", input_shape=INPUT_SHAPE))
        self.add(layers.Flatten())
        self.add(layers.Dense(1))
