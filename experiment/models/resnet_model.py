from tensorflow.keras.applications.resnet50 import ResNet50

from experiment.models.functional_model_wrapper import FunctionalModelWrapper

ResNetModel = FunctionalModelWrapper(ResNet50)
