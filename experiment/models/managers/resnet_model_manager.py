from keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.layers.base import Layer

from experiment.models.managers.model_manager import ModelManager


class ResNetModelManager(ModelManager):

    def get_model_constructor(self):
        return ResNet50

    @staticmethod
    def get_feature_layer(model: Model) -> Layer:
        return model.get_layer("conv5_block3_3_conv")

    def get_parameters(self):
        return self.get_model().layers[-1].get_weights()[0]
