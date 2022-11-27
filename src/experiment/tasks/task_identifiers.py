from enum import Enum

from src.experiment.tasks.calories_task import CaloriePredictionTask
from src.experiment.tasks.food_classification_task import FoodClassificationTask
from src.experiment.tasks.mass_task import MassPredictionTask
from src.experiment.tasks.test_task import TestTask


class Tasks(Enum):
    TEST = TestTask
    CALORIE = CaloriePredictionTask
    MASS = MassPredictionTask
    INGREDIENTS = FoodClassificationTask
