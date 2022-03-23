import warnings
from enum import Enum

from experiment.tasks.calories import CaloriePredictionTask
from experiment.tasks.category import FoodClassificationTask
from experiment.tasks.mass import MassPredictionTask
from experiment.tasks.task import BaseModel
from experiment.tasks.test import TestTask

warnings.filterwarnings("ignore")


class Tasks(Enum):
    TEST = TestTask
    CALORIE = CaloriePredictionTask
    MASS = MassPredictionTask
    INGREDIENTS = FoodClassificationTask


"""
Runner Settings
"""
task_selected = Tasks.MASS
base_model = BaseModel.TEST
n_epochs = 5  # while testing
"""
Gathers task, trains, and evaluates it
"""

if __name__ == "__main__":
    task = task_selected.value(base_model, n_epochs=n_epochs)
    task.train()
