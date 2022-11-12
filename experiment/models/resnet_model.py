from keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.layers.base import Layer

from experiment.models.functional_model_wrapper import FunctionalModelWrapper
from experiment.models.model_manager import ModelManager


class ResNetModelManager(ModelManager):
    def get_parameters(self):
        return list(self.model.fc.parameters())[0]

    def get_model_constructor(self):
        return FunctionalModelWrapper(ResNet50)

    def get_feature_layer(self, model: Model) -> Layer:
        return model.layer4[-1].conv3
