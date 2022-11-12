from enum import Enum

from experiment.models.ensemble_model import EnsembleModelManager
from experiment.models.resnet_model import ResNetModelManager
from experiment.models.test_model import TestModelManager
from experiment.models.vgg_model import VGGModelManager
from experiment.models.xception_model import XceptionModelManager


class ModelManagers(Enum):
    VGG = VGGModelManager
    RESNET = ResNetModelManager
    XCEPTION = XceptionModelManager
    TEST = TestModelManager
    ENSEMBLE = EnsembleModelManager


PRE_TRAINED_MODELS = [ModelManagers.VGG, ModelManagers.RESNET, ModelManagers.XCEPTION]
