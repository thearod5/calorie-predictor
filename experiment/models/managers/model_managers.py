from enum import Enum

from experiment.models.managers.ensemble_model_manager import EnsembleModelManager
from experiment.models.managers.resnet_model_manager import ResNetModelManager
from experiment.models.managers.test_model_manager import TestModelManager
from experiment.models.managers.vgg_model_manager import VGGModelManager
from experiment.models.managers.xception_model_manager import XceptionModelManager


class ModelManagers(Enum):
    VGG = VGGModelManager
    RESNET = ResNetModelManager
    XCEPTION = XceptionModelManager
    TEST = TestModelManager
    ENSEMBLE = EnsembleModelManager


PRE_TRAINED_MODELS = [ModelManagers.VGG, ModelManagers.RESNET, ModelManagers.XCEPTION]
