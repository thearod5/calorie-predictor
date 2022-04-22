from constants import INPUT_SHAPE
import tensorflow as tf
from experiment.models.test_model import TestModel
from experiment.models.ensemble_model import EnsembleModel


class BaseModel(Enum):
    VGG = "vgg"
    RESNET = "resnet"
    XCEPTION = "xception"
    TEST = "test"
    ENSEMBLE = "ensemble"


BASE_MODELS = {
    BaseModel.VGG.value: tf.keras.applications.VGG19,
    BaseModel.RESNET.value: tf.keras.applications.ResNet50,
    BaseModel.XCEPTION.value: tf.keras.applications.Xception,
    BaseModel.TEST.value: TestModel,
    BaseModel.ENSEMBLE.value: EnsembleModel

}

PRE_TRAINED_MODELS = [BaseModel.VGG.value, BaseModel.RESNET.value, BaseModel.XCEPTION.value]

def create_checkpoint_path(task_name, base_model_name):
    return os.path.join(PROJECT_DIR, "results", "checkpoints", task_name, base_model_name, "cp.ckpt")
