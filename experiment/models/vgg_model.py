from tensorflow.keras.applications.vgg19 import VGG19

from experiment.models.functional_model_wrapper import FunctionalModelWrapper

VGGModel = FunctionalModelWrapper(VGG19)
