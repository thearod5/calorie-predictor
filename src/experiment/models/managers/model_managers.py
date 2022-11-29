from enum import Enum

from src.experiment.models.managers.ensemble_model_manager import EnsembleModelManager
from src.experiment.models.managers.resnet_model_manager import ResNetModelManager
from src.experiment.models.managers.test_model_manager import TestModelManager
from src.experiment.models.managers.vgg_model_manager import VGGModelManager
from src.experiment.models.managers.xception_model_manager import XceptionModelManager


class ModelManagers(Enum):
    VGG = VGGModelManager
    RESNET = ResNetModelManager
    XCEPTION = XceptionModelManager
    TEST = TestModelManager
    ENSEMBLE = EnsembleModelManager


PRE_TRAINED_MODELS = [ModelManagers.VGG, ModelManagers.RESNET, ModelManagers.XCEPTION]
