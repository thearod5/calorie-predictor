import os

from constants import PROJECT_DIR


def get_checkpoint_path(task_name: str, base_model_name: str):
    return os.path.join(PROJECT_DIR, "results", "checkpoints", task_name, base_model_name, "cp.ckpt")
