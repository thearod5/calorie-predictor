from keras import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.python.layers.base import Layer

from experiment.models.functional_model_wrapper import FunctionalModelWrapper
from experiment.models.model_manager import ModelManager


class VGGModelManager(ModelManager):

    def get_model_constructor(self):
        return FunctionalModelWrapper(VGG19)

    def get_feature_layer(self, model: Model) -> Layer:
        return model.layers[-1]
