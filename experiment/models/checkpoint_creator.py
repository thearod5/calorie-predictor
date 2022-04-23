import os

from constants import PROJECT_DIR


def create_checkpoint_path(task_name, base_model_name):
    return os.path.join(PROJECT_DIR, "results", "checkpoints", task_name, base_model_name, "cp.ckpt")
