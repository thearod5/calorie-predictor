from keras.applications import Xception
from keras.models import Model
from tensorflow.python.layers.base import Layer

from src.experiment.models.managers.model_manager import ModelManager


class XceptionModelManager(ModelManager):
    def get_model_name(self) -> str:
        return "xception"

    def get_model_constructor(self):
        return Xception

    @staticmethod
    def get_feature_layer(model: Model) -> Layer:
        return model.get_layer("block14_sepconv2")
