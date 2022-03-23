import warnings
from enum import Enum

import tensorflow as tf

from experiment.tasks.calories import CaloriePredictionTask
from experiment.tasks.category import FoodClassificationTask
from experiment.tasks.mass import MassPredictionTask
from experiment.tasks.task import BaseModel
from experiment.tasks.test import TestTask

warnings.filterwarnings("ignore")


class Tasks(Enum):
    TEST = TestTask
    MASS = MassPredictionTask
    CALORIE = CaloriePredictionTask
    INGREDIENTS = FoodClassificationTask


"""
Runner Settings
"""
task_selected = Tasks.CALORIE
base_model = BaseModel.VGG
n_epochs = 5  # while testing
"""
Gathers task, trains, and evaluates it
"""

if __name__ == "__main__":
    task = task_selected.value(base_model, n_epochs=n_epochs)
    task.train()
