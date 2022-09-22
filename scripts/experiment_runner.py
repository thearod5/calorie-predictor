import argparse
import os
import sys
import warnings
from enum import Enum
from typing import Dict

# makes this runnable from command line
path_to_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(path_to_src)

from experiment.tasks.calories_task import CaloriePredictionTask
from experiment.tasks.classification_task import FoodClassificationTask
from experiment.tasks.mass_task import MassPredictionTask
from experiment.tasks.test_task import TestTask
from constants import N_EPOCHS

from experiment.tasks.base_task import BaseModel, Task, logger, set_data

warnings.filterwarnings("ignore")


class Tasks(Enum):
    TEST = TestTask
    CALORIE = CaloriePredictionTask
    MASS = MassPredictionTask
    INGREDIENTS = FoodClassificationTask


name2task: Dict[str, Tasks] = {
    "calories": Tasks.CALORIE,
    "mass": Tasks.MASS,
    "ingredients": Tasks.INGREDIENTS,
}


def get_args():
    parser = argparse.ArgumentParser(description='Compile a model for training or evaluation on some task.')
    parser.add_argument('data', choices=["test", "prod"])
    parser.add_argument('task', choices=name2task.keys())
    parser.add_argument('model', choices=[e.value for e in BaseModel])
    parser.add_argument('mode', choices=["train", "eval"], default="train")
    parser.add_argument('--dataset', default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data_env = args.data
    task_name = args.task
    base_model = args.model
    mode = args.mode

    # 2. Extract task and model
    set_data(data_env)
    task_selected: Tasks = name2task[task_name]

    # 3. Create task resources and train.
    task: Task = task_selected.value(base_model, n_epochs=N_EPOCHS)
    logger.info("Data Env: %s" % data_env)
    if mode == "train":
        task.train()
    elif mode == "eval":
        dataset_name = args.dataset
        task.eval(dataset_name)
    else:
        raise Exception("Unrecognized mode:" + mode)
