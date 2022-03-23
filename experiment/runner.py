from experiment.tasks.calories import CaloriePredictionTask
from experiment.tasks.mass import MassPredictionTask
from experiment.tasks.task import BaseModel
from experiment.tasks.test import TestTask

"""
Runner Settings
"""
task_name = "mass"
base_model = BaseModel.VGG
n_epochs = 5  # while testing
"""
Gathers task, trains, and evaluates it
"""
TASKS = {
    "test": TestTask,
    "mass": MassPredictionTask,
    "calorie": CaloriePredictionTask
}

if __name__ == "__main__":
    task = TASKS[task_name](base_model, n_epochs=n_epochs)
    task.train()
    task.evaluate()
