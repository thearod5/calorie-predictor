from enum import Enum

import tensorflow as tf
from keras import Input
from keras.layers import concatenate

from constants import ENSEMBLE_METHOD, INPUT_SHAPE, N_HIDDEN
from experiment.models.best_model_checkpoints import checkpoints

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
        models = [tf.keras.models.load_model(checkpoint) for checkpoint in checkpoints]
        if len(models) > 1:
            ensemble_method = ENSEMBLE_METHODS[ENSEMBLE_METHOD]
            model_output_layer = ensemble_method([model(input_layer) for model in models])
        else:
            model_output_layer = models.pop(0).input
        hidden_layer = tf.keras.layers.Dense(n_hidden)(model_output_layer)
        super().__init__(inputs=input_layer, outputs=hidden_layer)
