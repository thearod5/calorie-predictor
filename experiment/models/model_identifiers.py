from enum import Enum

from experiment.models.ensemble_model import EnsembleModel
from experiment.models.resnet_model import ResNetModel
from experiment.models.test_model import TestModel
from experiment.models.vgg_model import VGGModel
from experiment.models.xception_model import XceptionModel


class BaseModel(Enum):
    VGG = VGGModel
    RESNET = ResNetModel
    XCEPTION = XceptionModel
    TEST = TestModel
    ENSEMBLE = EnsembleModel


base = {
    "VGG": BaseModel.VGG,
    "RESNET": BaseModel.RESNET,
    "XCEPTION": BaseModel.XCEPTION,
    "TEST": TestModel,
    "ENSEMBLE": EnsembleModel
}
PRE_TRAINED_MODELS = [BaseModel.VGG, BaseModel.RESNET, BaseModel.XCEPTION]
