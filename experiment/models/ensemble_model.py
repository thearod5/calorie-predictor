from enum import Enum

import tensorflow as tf
from keras import Input
from keras.layers import concatenate

from constants import ENSEMBLE_METHOD, INPUT_SHAPE, N_HIDDEN
from experiment.models.best_model_checkpoints import checkpoints


class EnsembleMethod(Enum):
    CONCATENATE = "concatenate"


ENSEMBLE_METHODS = {
    EnsembleMethod.CONCATENATE.value: concatenate
}


class EnsembleModel(tf.keras.Model):
    def __init__(self, n_hidden=N_HIDDEN):
        """
        Represents a combination of multiple models
        :param n_hidden: number of hidden layers to use
        """
        models = [tf.keras.models.load_model(checkpoint) for checkpoint in checkpoints]
        if len(models) > 1:
            input_layer = Input(shape=INPUT_SHAPE, name="shared_input")
            ensemble_method = ENSEMBLE_METHODS[ENSEMBLE_METHOD]
            model_output_layer = ensemble_method([model(input_layer) for model in models])
        else:
            single_model = models.pop(0)
            input_layer = single_model.input
            model_output_layer = single_model.output
        hidden_layer = tf.keras.layers.Dense(n_hidden)(model_output_layer)
        super().__init__(inputs=input_layer, outputs=hidden_layer)
