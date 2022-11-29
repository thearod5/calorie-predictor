import argparse
import os
import sys
import warnings
from enum import Enum

from src.experiment.cam.cam_loss_alpha import AlphaStrategy
from src.experiment.models.checkpoint_creator import get_checkpoint_path
from src.experiment.tasks.task_identifiers import Tasks

# makes this runnable from command line

path_to_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(path_to_src)

from constants import N_EPOCHS, get_data_dir
from src.experiment.models.managers.model_managers import ModelManagers
from src.experiment.tasks.base_task import AbstractTask, logger, set_data
import tensorflow as tf
from src.experiment.tasks.calories_task import CaloriePredictionTask

warnings.filterwarnings("ignore")


def name2enum(name: str, enum_class: Enum):
    for e in enum_class:
        if e.name == name.upper():
            return e
    raise Exception("Unrecognized %s %s:" % (enum_class.__class__.__name__, name))


def get_args():
    parser = argparse.ArgumentParser(description='Compile a model for training or evaluation on some task.')

    parser.add_argument('--model', choices=[e.name.lower() for e in ModelManagers])
    parser.add_argument('--path', default=None)
    parser.add_argument('--mode', choices=["train", "eval", "traineval"], default="traineval")
    parser.add_argument('--alpha', choices=[e.name.lower() for e in AlphaStrategy], default=AlphaStrategy.ADAPTIVE.name)
    parser.add_argument('--checkpoint', default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data_env = "prod"

    print("-" * 25)
    print(args)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # 2. Extract task and model
    set_data(data_env)
    data_dir = get_data_dir()


    def create_task() -> CaloriePredictionTask:
        model_manager_params = {}
        if args.model:
            model_name = args.model
            checkpoint_name = None
        else:
            assert args.path is not None, "Expected model or model_path to be defined."
            assert "-" in args.path
            task_name, model_name = args.path.split("-")
            checkpoint_task_class: AbstractTask = name2enum(task_name, Tasks).value
            checkpoint_task_name = checkpoint_task_class.__name__
            model_manager_params["model_path"] = get_checkpoint_path(checkpoint_task_name, model_name)
            checkpoint_name = os.path.join("pretrain", task_name)
        model_manager_class: ModelManagers = name2enum(model_name, ModelManagers)
        model_manager = model_manager_class.value(**model_manager_params)
        return CaloriePredictionTask(model_manager, n_epochs=N_EPOCHS, use_cam=True, checkpoint_name=checkpoint_name)


    # 4. Train then evaluate
    logger.info("Data Env: %s" % data_env)
    if "train" in args.mode:
        alpha_strategy: AlphaStrategy = name2enum(args.alpha, AlphaStrategy)
        task = create_task()
        task.train(alpha_strategy=alpha_strategy)
    if "eval" in args.mode:
        task = create_task()
        task.eval()
