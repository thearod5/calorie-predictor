import argparse
import os
import sys
import warnings
from enum import Enum

# makes this runnable from command line
from experiment.tasks.calories_task import CaloriePredictionTask

path_to_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(path_to_src)

from constants import N_EPOCHS, get_data_dir
from experiment.models.managers.model_managers import ModelManagers
from experiment.tasks.base_task import logger, set_data, AbstractTask
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
    parser.add_argument('model', choices=[e.name.lower() for e in ModelManagers])
    parser.add_argument('mode', choices=["train", "eval"], default="train")
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--cam', dest='cam', default=True, action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data_env = args.data
    task_name = args.task
    model_manager_name = args.model
    mode = args.mode
    use_cam = args.cam

    print("-" * 25)
    print(args)

    # 2. Extract task and model
    set_data(data_env)
    data_dir = get_data_dir()
    task_selected: Tasks = name2enum(task_name, Tasks)
    model_manager_class: ModelManagers = name2enum(model_manager_name, ModelManagers)

    # 3. Create task resources and train.
    kwargs = {}
    if use_cam:
        kwargs["use_cam"] = True
    model_manager = model_manager_class.value()
    task: AbstractTask = task_selected.value(model_manager, n_epochs=N_EPOCHS, **kwargs)
    logger.info("Data Env: %s" % data_env)
    if mode == "train":
        task.train()
    elif mode == "eval":
        dataset_name = args.dataset
        task.eval(dataset_name)
    else:
        raise Exception("Unrecognized mode:" + mode)
