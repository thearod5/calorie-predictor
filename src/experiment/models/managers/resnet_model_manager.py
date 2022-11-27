from keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.layers.base import Layer

from src.experiment.models.managers.model_manager import ModelManager


class ResNetModelManager(ModelManager):

    def get_model_name(self) -> str:
        return "resnet"

    def get_model_constructor(self):
        return ResNet50

    @staticmethod
    def get_feature_layer(model: Model) -> Layer:
        return model.get_layer("conv5_block3_3_conv")

    def get_feature_weights(self):
        return self.get_model().layers[-1].get_weights()[0]
