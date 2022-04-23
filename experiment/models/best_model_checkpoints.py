from constants import BEST_CLASSIFICATION_MODEL, BEST_MASS_MODEL
from experiment.models.checkpoint_creator import create_checkpoint_path
from experiment.models.ensemble_model import FOOD_CLASSIFICATION_TASK, MASS_PREDICTION_TASK

checkpoints = [create_checkpoint_path(FOOD_CLASSIFICATION_TASK, BEST_CLASSIFICATION_MODEL),
               create_checkpoint_path(MASS_PREDICTION_TASK, BEST_MASS_MODEL)]
