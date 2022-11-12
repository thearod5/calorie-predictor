from keras import Model
from tensorflow.keras.applications.xception import Xception
from tensorflow.python.layers.base import Layer

from experiment.models.functional_model_wrapper import FunctionalModelWrapper
from experiment.models.model_manager import ModelManager


class XceptionModelManager(ModelManager):
    def get_parameters(self):
        return list(self.model.last_linear.parameters())[0]

    def get_model_constructor(self):
        return FunctionalModelWrapper(Xception)

    def get_feature_layer(self, model: Model) -> Layer:
        return model.layers.conv4
