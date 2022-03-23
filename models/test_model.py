"""
A models for basic testing of inputs and outputs.
"""
from tensorflow.keras import layers, models

from constants import INPUT_SIZE

N_FEATURES = 32
KERNEL_SIZE = (3, 3)

test_model = models.Sequential()
test_model.add(layers.Conv2D(N_FEATURES, KERNEL_SIZE, activation="relu", input_shape=INPUT_SIZE))
test_model.add(layers.Flatten())
test_model.add(layers.Dense(1))
