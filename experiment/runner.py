import argparse
import os
import sys
# makes this runnable from command line
from typing import Dict

from experiment.tasks.classification_sample_builder import ClassificationSubsetTask

path_to_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(path_to_src)

import warnings
from enum import Enum

from constants import N_EPOCHS, set_data

from experiment.tasks.calories_task import CaloriePredictionTask
from experiment.tasks.classification_task import FoodClassificationTask
from experiment.tasks.mass_task import MassPredictionTask
from experiment.tasks.base_task import BaseModel, Task
from experiment.tasks.test_task import TestTask

warnings.filterwarnings("ignore")


class Tasks(Enum):
    TEST = TestTask
    CALORIE = CaloriePredictionTask
    MASS = MassPredictionTask
    INGREDIENTS = FoodClassificationTask
    INGREDIENTS_SAMPLE = ClassificationSubsetTask


name2task: Dict[str, Tasks] = {
    "calories": Tasks.CALORIE,
    "mass": Tasks.MASS,
    "ingredients": Tasks.INGREDIENTS,
    "ingredients-sample": Tasks.INGREDIENTS_SAMPLE
}

name2model = {
    "vgg": BaseModel.VGG,
    "resnet": BaseModel.RESNET,
    "xception": BaseModel.XCEPTION,
    "test": BaseModel.TEST
}


def get_args():
    parser = argparse.ArgumentParser(description='Compile a model for training or evaluation on some task.')
    parser.add_argument('data', choices=["test", "prod"])
    parser.add_argument('task', choices=name2task.keys())
    parser.add_argument('model', choices=["vgg", "resnet", "xception"])
    parser.add_argument('mode', choices=["train", "eval"], default="train")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data_env = args.data
    task_name = args.task
    model = args.model
    mode = args.mode

    # 2. Extract task and model
    set_data(data_env)
    task_selected: Tasks = name2task[task_name]

    base_model = name2model[model]

    # 3. Create task resources and train.
    task: Task = task_selected.value(base_model, n_epochs=N_EPOCHS)

    if mode == "train":
        task.train()
    elif mode == "eval":
        task.eval()
    else:
        raise Exception("Unrecognized mode:" + mode)
