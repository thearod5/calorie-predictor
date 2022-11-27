from enum import Enum

import tensorflow as tf
from keras import Input
from keras.layers import concatenate
from keras.models import Model
from tensorflow.python.layers.base import Layer

from constants import ENSEMBLE_METHOD, INPUT_SHAPE, N_HIDDEN
from src.experiment.models.best_model_checkpoints import checkpoints
from src.experiment.models.managers.model_manager import ModelManager


class EnsembleMethod(Enum):
    CONCATENATE = "concatenate"


ENSEMBLE_METHODS = {
    EnsembleMethod.CONCATENATE.value: concatenate
}


class EnsembleModelManager(ModelManager):
    def get_model_name(self) -> str:
        return "ensemble"

    def __init__(self, n_hidden=N_HIDDEN):
        """
        Represents a combination of multiple models
        :param n_hidden: number of hidden layers to use
        """
        super().__init__()
        self.n_hidden = n_hidden

    def get_model_constructor(self):
        models = [tf.keras.models.load_model(checkpoint) for checkpoint in checkpoints]
        if len(models) > 1:
            input_layer = Input(shape=INPUT_SHAPE, name="shared_input")
            ensemble_method = ENSEMBLE_METHODS[ENSEMBLE_METHOD]
            model_output_layer = ensemble_method([model(input_layer) for model in models])
        else:
            single_model = models.pop(0)
            input_layer = single_model.input
            model_output_layer = single_model.output
        hidden_layer = tf.keras.layers.Dense(self.n_hidden)(model_output_layer)
        model = Model(inputs=input_layer, outputs=hidden_layer)
        return lambda: model

    @staticmethod
    def get_feature_layer(model: Model) -> Layer:
        raise NotImplementedError()

    def get_feature_weights(self):
        raise NotImplementedError()
