from tensorflow.keras.applications.xception import Xception

from experiment.models.functional_model_wrapper import FunctionalModelWrapper

XceptionModel = FunctionalModelWrapper(Xception)
