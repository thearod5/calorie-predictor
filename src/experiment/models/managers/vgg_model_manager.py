from keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.python.layers.base import Layer

from src.experiment.models.functional_model_wrapper import FunctionalModelWrapper
from src.experiment.models.managers.model_manager import ModelManager


class VGGModelManager(ModelManager):
    def get_model_name(self) -> str:
        return "vgg"

    def get_model_constructor(self):
        return FunctionalModelWrapper(VGG19)

    @staticmethod
    def get_feature_layer(model: Model) -> Layer:
        return model.layers[-1]

    def get_feature_weights(self):
        raise NotImplementedError()
