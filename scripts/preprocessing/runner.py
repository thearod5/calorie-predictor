import os

from constants import PROJECT_DIR
from scripts.preprocessing.ecustfd import EcustfdProcessor
from scripts.preprocessing.food_image import FoodImageProcessor
from scripts.preprocessing.menu_match import MenuMatchPrecessor
from scripts.preprocessing.nutrition5k import Nutrition5kH264Images, Nutrition5kJpgImages
from scripts.preprocessing.processor import ProcessingSettings
from scripts.preprocessing.unimib2016 import UNIMIB2016Processor

"""
The following script will read, resize, and save files from all of the experiment datasets.
The dataset consists of two file formats: .jpg and .h264
Note, only the first frame of the .h264 video is taken as the image.
All images are resized to 224x224 and saved in .jpg
"""


def process_nutrition5k(settings: ProcessingSettings):
    jpg_images = Nutrition5kJpgImages()
    h264_images = Nutrition5kH264Images()
    jpg_images.process(settings)
    h264_images.process(settings)


PATH_TO_PROJECT = "/Volumes/Betito HDD/Datasets/calorie-predictor"
PATH_TO_OUTPUT_DIR = os.path.join(PROJECT_DIR, "processed")
IMAGE_NAME_SEPARATOR = "-"

if __name__ == "__main__":
    """
    Runtime Variables
    """

    SHOW_ERRORS = False
    THROW_ERROR = False
    settings = ProcessingSettings(SHOW_ERRORS, THROW_ERROR)
    """
    Processing 
    """
    process_nutrition5k(settings)
    EcustfdProcessor().process(settings)
    MenuMatchPrecessor().process(settings)
    UNIMIB2016Processor().process(settings)
    FoodImageProcessor().process(settings).create_output_file()  # .process(settings)
