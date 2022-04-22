import tensorflow as tf
from keras import Input
from keras.layers import concatenate
from enum import Enum

from constants import BEST_CLASSIFICATION_MODEL, BEST_MASS_MODEL, INPUT_SHAPE, N_HIDDEN, ENSEMBLE_METHOD
from experiment.models.checkpoint_creator import create_checkpoint_path

FOOD_CLASSIFICATION_TASK = "FoodClassificationTask"
MASS_PREDICTION_TASK = "MassPredictionTask"


class EnsembleMethod(Enum):
    CONCATENATE = "concatenate"


ENSEMBLE_METHODS = {
    EnsembleMethod.CONCATENATE: concatenate
}


class EnsembleModel(tf.keras.Model):
    def __init__(self, n_hidden=N_HIDDEN):
        input_layer = Input(shape=INPUT_SHAPE, name="shared_input")
        classification_model = tf.keras.models.load_model(
            create_checkpoint_path(FOOD_CLASSIFICATION_TASK, BEST_CLASSIFICATION_MODEL))
        mass_model = tf.keras.models.load_model(create_checkpoint_path(MASS_PREDICTION_TASK, BEST_MASS_MODEL))
        ensemble_method = ENSEMBLE_METHODS[ENSEMBLE_METHOD]
        combined_layer = ensemble_method([classification_model(input_layer),
                                          mass_model(input_layer)])
        hidden_layer = tf.keras.layers.Dense(n_hidden)(combined_layer)
        super().__init__(inputs=input_layer, outputs=hidden_layer)
