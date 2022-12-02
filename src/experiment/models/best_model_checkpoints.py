from constants import BEST_CLASSIFICATION_MODEL, BEST_MASS_MODEL
from src.experiment.models.checkpoint_creator import get_checkpoint_path

FOOD_CLASSIFICATION_TASK = "FoodClassificationTask"
MASS_PREDICTION_TASK = "MassPredictionTask"

checkpoints = [get_checkpoint_path(MASS_PREDICTION_TASK, BEST_MASS_MODEL),
               get_checkpoint_path(FOOD_CLASSIFICATION_TASK, BEST_CLASSIFICATION_MODEL)]
