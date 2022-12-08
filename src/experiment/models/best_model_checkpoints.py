import os

mass_model_checkpoint = os.path.expanduser(os.path.join("~", "models", "mass", "resnet"))
cam_model_checkpoint = os.path.expanduser(os.path.join("~", "models", "cam", "xception"))

checkpoints = [mass_model_checkpoint,
               cam_model_checkpoint]
