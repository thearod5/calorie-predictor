from constants import BEST_MASS_MODEL
from experiment.models.checkpoint_creator import get_checkpoint_path

FOOD_CLASSIFICATION_TASK = "FoodClassificationTask"
MASS_PREDICTION_TASK = "MassPredictionTask"

checkpoints = [get_checkpoint_path(MASS_PREDICTION_TASK, BEST_MASS_MODEL)]
