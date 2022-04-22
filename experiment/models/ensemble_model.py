from constants import BEST_CLASSIFICATION_MODEL, BEST_MASS_MODEL, ENSEMBLE_METHOD_NAME, N_HIDDEN
from experiment.tasks.mass_task import MassPredictionTask
from experiment.models.base_models import BaseModel, BASE_MODELS, create_checkpoint_path
import tensorflow as tf

FOOD_CLASSIFICATION_TASK = "FoodClassificationTask"
MASS_PREDICTION_TASK = "MassPredictionTask"


class EnsembleModel(tf.keras.Model):
    def __init__(self, n_hidden=N_HIDDEN):
        classification_model = tf.keras.models.load_model(create_checkpoint_path(FOOD_CLASSIFICATION_TASK, BEST_CLASSIFICATION_MODEL))
        mass_model = tf.keras.models.load_model(create_checkpoint_path(MASS_PREDICTION_TASK, BEST_MASS_MODEL))
        pre_trained_models = [classification_model, mass_model]
        input_layers = [model.input for model in pre_trained_models]
        output_layers = [model.layers[-2] for model in pre_trained_models]
        super().__init__(inputs=input_layers, outputs=output_layers)
        hidden_layer = tf.keras.layers.Dense(n_hidden)
        self.add(hidden_layer)

