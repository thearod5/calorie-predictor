from scripts.preprocessing.food_image import FoodImageProcessor
from scripts.preprocessing.nutrition5k import Nutrition5kH264Images, Nutrition5kJpgImages
from scripts.preprocessing.processor import ProcessingSettings

"""
The following script will read, resize, and save files from the nutrition5k dataset.
The dataset consists of two file formats: .jpg and .h264
All images are resized to 224x224 and saved in .jpg
"""


def process_nutrition5k(settings: ProcessingSettings):
    jpg_images = Nutrition5kJpgImages()
    h264_images = Nutrition5kH264Images()
    jpg_images.process(settings)
    h264_images.process(settings)


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
    # process_nutrition5k(settings)
    # EcustfdProcessor().process(settings)
    # MenuMatchPrecessor().process(settings)
    # UNIMIB2016Processor().process(settings)
    FoodImageProcessor().process(settings).create_output_file()  # .process(settings)
