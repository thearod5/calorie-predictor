import os

from constants import CHECKPOINT_BASE_PATH


def get_checkpoint_path(task_name: str, base_model_name: str, checkpoint_name: str = None):
    task_checkpoint_path = os.path.join(CHECKPOINT_BASE_PATH, task_name)
    if checkpoint_name:
        task_checkpoint_path = os.path.join(task_checkpoint_path, checkpoint_name)
    return os.path.join(task_checkpoint_path, base_model_name, "cp.ckpt")
