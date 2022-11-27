from keras.models import Model
from tensorflow.keras.applications.xception import Xception
from tensorflow.python.layers.base import Layer

from src.experiment.models.functional_model_wrapper import FunctionalModelWrapper
from src.experiment.models.managers.model_manager import ModelManager


class XceptionModelManager(ModelManager):
    def get_model_name(self) -> str:
        return "xception"

    def get_model_constructor(self):
        return FunctionalModelWrapper(Xception)

    @staticmethod
    def get_feature_layer(model: Model) -> Layer:
        return model.layers.conv4

    def get_feature_weights(self):
        return list(self.get_model().last_linear.parameters())[0]
