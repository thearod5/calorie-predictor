import os

from constants import CHECKPOINT_BASE_PATH

mass_model_checkpoint = os.path.join(CHECKPOINT_BASE_PATH, "pretrain", "mass", "xception")
cam_model_checkpoint = os.path.join(CHECKPOINT_BASE_PATH, "rq1", "cam-simplified", "xception")

checkpoints = [mass_model_checkpoint,
               cam_model_checkpoint]
