import os
from os.path import *

PROJECT_DIR = dirname(abspath(__file__))
ENV = "test"  # dev | prod

RANDOM_SEED = 0
N_CHANNELS = 3
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = IMAGE_SIZE + (N_CHANNELS,)
BATCH_SIZE = 32
EXT_SEP = "."
TEST_SPLIT_SIZE = .20
N_EPOCHS = 10
LOG_SKIPPED_ENTRIES = False


def get_data_dir():
    if ENV == "test":
        return os.path.join(PROJECT_DIR, 'data')
    elif ENV == "dev":
        return "/Volumes/Betito HDD/Datasets/calorie-predictor/processed"
    else:
        return os.path.join(PROJECT_DIR, "processed")


def set_env(env: str):
    global ENV
    ENV = env
