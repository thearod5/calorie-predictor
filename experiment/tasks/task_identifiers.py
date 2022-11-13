from enum import Enum

from experiment.tasks.calories_task import CaloriePredictionTask
from experiment.tasks.food_classification_task import FoodClassificationTask
from experiment.tasks.mass_task import MassPredictionTask
from experiment.tasks.test_task import TestTask


class Tasks(Enum):
    TEST = TestTask
    CALORIE = CaloriePredictionTask
    MASS = MassPredictionTask
    INGREDIENTS = FoodClassificationTask
