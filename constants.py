import os
from os.path import *

PROJECT_DIR = dirname(abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RANDOM_SEED = 0
N_CHANNELS = 3
IMAGE_SIZE = (224, 224)
INPUT_SIZE = IMAGE_SIZE + (N_CHANNELS,)
BATCH_SIZE = 32
EXT_SEP = "."
TEST_SPLIT_SIZE = .25
N_EPOCHS = 50
