import os
import sys

# makes this runnable from command line
path_to_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(path_to_src)

import warnings
from enum import Enum

from experiment.tasks.calorie_task import CaloriePredictionTask
from experiment.tasks.category_task import FoodClassificationTask
from experiment.tasks.mass_task import MassPredictionTask
from experiment.tasks.base_task import BaseModel
from experiment.tasks.test_task import TestTask

warnings.filterwarnings("ignore")


class Tasks(Enum):
    TEST = TestTask
    CALORIE = CaloriePredictionTask
    MASS = MassPredictionTask
    INGREDIENTS = FoodClassificationTask


name2task = {
    "calories": Tasks.CALORIE,
    "mass": Tasks.MASS,
}

name2model = {
    "vgg": BaseModel.VGG,
    "resnet": BaseModel.RESNET,
    "xception": BaseModel.XCEPTION
}
"""
Runner Settings
"""

n_epochs = 10  # while testing
"""
Gathers task, trains, and evaluates it
"""

if __name__ == "__main__":

    # 1. Process arguments
    if len(sys.argv) != 3:
        raise Exception("Expected: [TaskName] [ModelName]")
    task, model = sys.argv[1:]

    # 2. Extract task and model
    task_selected = name2task[task]
    base_model = name2model[model]

    # 3. Create task resources and train.
    task = task_selected.value(base_model, n_epochs=n_epochs)
    task.train()
