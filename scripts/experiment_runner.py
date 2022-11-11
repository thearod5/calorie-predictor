import argparse
import os
import sys
import warnings
from enum import Enum

# makes this runnable from command line
path_to_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(path_to_src)

from experiment.tasks.food_classification_task import FoodClassificationTask
from experiment.tasks.calories_task import CaloriePredictionTask
from experiment.tasks.mass_task import MassPredictionTask
from experiment.tasks.test_task import TestTask
from constants import N_EPOCHS
from experiment.models.model_identifiers import BaseModel
from experiment.tasks.base_task import logger, set_data, BaseTask
from experiment.tasks.task_identifiers import Tasks

warnings.filterwarnings("ignore")


def name2enum(name: str, enum_class: Enum):
    for e in enum_class:
        if e.name == name.upper():
            return e
    raise Exception("Unrecognized %s %s:" % (enum_class.__class__.__name__, name))


def get_args():
    parser = argparse.ArgumentParser(description='Compile a model for training or evaluation on some task.')
    parser.add_argument('data', choices=["test", "prod"])
    parser.add_argument('task', choices=[e.name.lower() for e in Tasks])
    parser.add_argument('model', choices=[e.name.lower() for e in BaseModel])
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
    task_selected: Tasks = name2enum(task_name, Tasks)
    model_selected: BaseModel = name2enum(base_model, BaseModel)

    # 3. Create task resources and train.
    task: BaseTask = task_selected.value(model_selected, n_epochs=N_EPOCHS)
    logger.info("Data Env: %s" % data_env)
    if mode == "train":
        task.train()
    elif mode == "eval":
        dataset_name = args.dataset
        task.eval(dataset_name)
    else:
        raise Exception("Unrecognized mode:" + mode)
