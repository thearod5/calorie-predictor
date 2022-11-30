import argparse
import os
import sys
import warnings
from enum import Enum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# makes this runnable from command line
path_to_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(path_to_src)
from logging_utils.utils import format_header
from src.experiment.trainers.cam.cam_loss_alpha import AlphaStrategy
from constants import CHECKPOINT_BASE_PATH, N_EPOCHS, get_data_dir
from src.experiment.models.managers.model_managers import ModelManagers
from src.experiment.tasks.base_task import AbstractTask, set_data
from src.experiment.tasks.calories_task import CaloriePredictionTask
from src.experiment.models.checkpoint_creator import get_checkpoint_path
from src.experiment.tasks.task_identifiers import Tasks

warnings.filterwarnings("ignore")


def name2enum(name: str, enum_class: Enum):
    for e in enum_class:
        if e.name == name.upper():
            return e
    raise Exception("Unrecognized %s %s:" % (enum_class.__class__.__name__, name))


def get_args():
    parser = argparse.ArgumentParser(description='Compile a model for training or evaluation on some task.')

    parser.add_argument('--model', choices=[e.name.lower() for e in ModelManagers])  # The model to create from scratch
    parser.add_argument('--load', default=None)
    parser.add_argument('--type', default="resnet")
    parser.add_argument('--pretrain', default=None)
    parser.add_argument('--mode', choices=["train", "eval"], default="train")
    parser.add_argument('--alpha', choices=[e.name.lower() for e in AlphaStrategy], default=AlphaStrategy.CONSTANT.name)
    parser.add_argument('--checkpoint', default=None)

    return parser.parse_args()


if __name__ == "__main__":
    # 1. Create argument and set data env
    args = get_args()
    data_env = "prod"
    set_data(data_env)
    data_dir = get_data_dir()
    n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))

    # 2. Log information
    print(format_header("Cam Runner"))
    print("Arguments:", repr(args))
    print("Num GPUs Available: " + str(n_gpus))
    print("Data Env: %s" % data_env)


    def create_task() -> CaloriePredictionTask:
        model_manager_params = {}

        if args.load:
            assert args.model is not None, "Expected model to be defined (e.g. --model resnet)."
            model_name = args.type
            model_manager_params["model_path"] = os.path.join(CHECKPOINT_BASE_PATH, args.load)
            model_manager_params["export_path"] = model_manager_params["model_path"]
            checkpoint_name = "STOP"
        elif args.model:
            model_name = args.model
            checkpoint_name = "baseline"
            print("Creating new model:", model_name)
        else:
            assert args.pretrain is not None, "Expected model or pretrain to be defined."
            assert "-" in args.pretrain
            task_name, model_name = args.pretrain.split("-")
            checkpoint_task_class: AbstractTask = name2enum(task_name, Tasks).value
            checkpoint_task_name = checkpoint_task_class.__name__
            model_manager_params["model_path"] = get_checkpoint_path(checkpoint_task_name, model_name)
            model_manager_params["export_path"] = get_checkpoint_path("CamTrainer", task_name,
                                                                      checkpoint_name="pretrain")
            checkpoint_name = os.path.join("pretrain", task_name)
            print("Loading pre-trained model on task (%s) with model (%s)." % (task_name, model_name))

        if model_manager_params.get("model_path", None):
            temp_path = model_manager_params["model_path"]
            assert os.path.exists(temp_path), "Model path does not exists: " + temp_path

        model_manager_class: ModelManagers = name2enum(model_name, ModelManagers)
        model_manager = model_manager_class.value(**model_manager_params)
        return CaloriePredictionTask(model_manager, n_epochs=N_EPOCHS, use_cam=True, checkpoint_name=checkpoint_name)


    # 3. Create task and arguments
    task = create_task()
    alpha_strategy: AlphaStrategy = name2enum(args.alpha, AlphaStrategy)

    # 4. Train then evaluate
    if "train" in args.mode:
        print("Alpha Strategy:", alpha_strategy)
        task.train(alpha_strategy=alpha_strategy)
    if "eval" in args.mode:
        task.eval()
