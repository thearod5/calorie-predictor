import os
from os.path import *

from dotenv import load_dotenv

load_dotenv()
ENV = os.getenv("ENV", "prod")

PROJECT_DIR = dirname(abspath(__file__))
PATH_TO_PROJECT = "/Volumes/Betito HDD/Datasets/calorie-predictor"
PATH_TO_OUTPUT_DIR = os.path.join(PROJECT_DIR, "processed")
IMAGE_NAME_SEPARATOR = "-"
IMAGE_DIR = "images"
CHECKPOINT_BASE_PATH = os.path.join(PROJECT_DIR, "results", "checkpoints")

RANDOM_SEED = 0
N_CHANNELS = 3
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = IMAGE_SIZE + (N_CHANNELS,)
BATCH_SIZE = 32
EXT_SEP = "."
TEST_SPLIT_SIZE = .20
N_EPOCHS = 50
LOG_SKIPPED_ENTRIES = False
MAXIMUM_BUFFER_SIZE = 20000
LOG_CONFIG_FILE = os.path.join(PROJECT_DIR, 'src/logging_utils', 'logging.conf')
LOG_CHAR = "*"
HEADING_CHARS = LOG_CHAR * 20
SUBHEADING_CHARS = "-" * 10
CLASSIFICATION_DATASETS = ["unimib", "food_images"]
BEST_MASS_MODEL = 'resnet'
BEST_CLASSIFICATION_MODEL = 'xception'
ENSEMBLE_METHOD = 'concatenate'
N_HIDDEN = 100


def get_data_dir():
    if ENV == "test":
        return os.path.join(PROJECT_DIR, 'data')
    elif ENV == "prod":
        return os.path.join(PROJECT_DIR, "processed")
    else:
        raise Exception("Unknown data environment:" + ENV)


def set_data(test_env: str):
    global ENV
    ENV = test_env


def get_cam_path():
    return os.path.join(PROJECT_DIR, "processed", "cam")


DEFAULT_TURK_RESULTS = os.path.join(PROJECT_DIR, "src/turk_task", "data")
