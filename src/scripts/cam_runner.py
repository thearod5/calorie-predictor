import argparse
import os
import sys
import warnings
from enum import Enum
from typing import Type

from src.experiment.tasks.base_task import AbstractTask
from src.experiment.tasks.task_identifiers import Tasks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# makes this runnable from command line
path_to_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(path_to_src)
from src.logging_utils.utils import format_header
from src.experiment.models.managers.model_manager import ModelManager
from src.experiment.trainers.cam.cam_loss_alpha import AlphaStrategy
from constants import CHECKPOINT_BASE_PATH, ENV, N_EPOCHS, get_data_dir
from src.experiment.models.managers.model_managers import ModelManagers

warnings.filterwarnings("ignore")


class ModelState(Enum):
    NEW = "new",
    LOAD = "load",


class TaskJob(Enum):
    TRAIN = "train"
    EVAL = "eval"


class Args:
    job: TaskJob
    model_state: ModelState
    model_manager_class: Type[ModelManager]
    base_path: str
    export_path: str
    alpha_strategy: AlphaStrategy
    task: Tasks

    def __init__(self):
        parser = argparse.ArgumentParser(description='Compile a model for training or evaluation on some task.')

        parser.add_argument('job', choices=[e.name.lower() for e in TaskJob])
        parser.add_argument('state', choices=[e.name.lower() for e in ModelState])
        parser.add_argument('manager',
                            choices=[e.name.lower() for e in ModelManagers])  # The model to create from scratch
        parser.add_argument('--path', default=None)
        parser.add_argument('--modify', default=False, action="store_true")
        parser.add_argument('--export', default=None)
        parser.add_argument('--nocam', default=False, action="store_true")
        parser.add_argument('--project', default=CHECKPOINT_BASE_PATH)
        parser.add_argument('--alpha', choices=[e.name.lower() for e in AlphaStrategy],
                            default=AlphaStrategy.CONSTANT_BALANCED.name)
        parser.add_argument('--task', choices=[t.name.lower() for t in Tasks], default=Tasks.CALORIE.name)
        args = parser.parse_args()
        self.job = self.name2enum(args.job, TaskJob)
        self.model_state = self.name2enum(args.state, ModelState)
        self.model_manager_class = self.name2enum(args.manager, ModelManagers).value
        self.base_path = os.path.join(args.project, args.path) if args.path else None
        self.export_path = os.path.join(args.project, args.export) if args.export else None
        self.alpha_strategy = self.name2enum(args.alpha, AlphaStrategy)
        self.modify = args.modify
        self.use_cam = not args.nocam
        self.task: Type[AbstractTask] = self.name2enum(args.task, Tasks).value

    @staticmethod
    def name2enum(name: str, enum_class):
        for e in enum_class:
            if e.name == name.upper():
                return e
        raise Exception("Unrecognized %s %s:" % (enum_class.__class__.__name__, name))


def create_model_args(model_state: ModelState, base_path: str, export: str, modify: bool):
    if model_state == ModelState.NEW:
        assert export is not None, "Expected export path to be defined."
        return {
            "base_model_path": None,
            "export_path": export,
            "create_task_model": True
        }

    if model_state == ModelState.LOAD:
        assert base_path is not None, "Expected base_path to be defined."
        return {
            "base_model_path": base_path,
            "export_path": export if export else base_path,
            "create_task_model": modify
        }


if __name__ == "__main__":
    # 1. Create argument and set data env
    runner_args = Args()
    data_dir = get_data_dir()
    n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))

    # 2. Log information
    print(format_header("Cam Runner"))
    print("Arguments:", repr(runner_args.__dict__))
    print("Num GPUs Available: " + str(n_gpus))
    print("Data Env: %s" % ENV)

    # 3. Create task and train arguments
    train_kwargs = {}
    task_kwargs = {}
    model_args = create_model_args(runner_args.model_state,
                                   runner_args.base_path,
                                   runner_args.export_path,
                                   runner_args.modify)
    model_manager = runner_args.model_manager_class(**model_args)

    # 4. Create task
    if runner_args.task == Tasks.CALORIE.value:
        train_kwargs["alpha_strategy"] = runner_args.alpha_strategy
        task_kwargs["use_cam"] = runner_args.use_cam
    task = runner_args.task(model_manager, n_epochs=N_EPOCHS, **task_kwargs)

    if runner_args.job == TaskJob.TRAIN:
        task.train(**train_kwargs)
    if runner_args.job == TaskJob.EVAL:
        if runner_args.use_cam and runner_args.task == Tasks.CALORIE.value:
            task.trainer.perform_evaluation(task.get_test_data())
        else:
            task.eval()
