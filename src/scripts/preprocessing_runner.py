import tensorflow as tf

from src.preprocessing.base_processor import ProcessingSettings
from src.preprocessing.ecustfd_processor import EcustfdProcessor
from src.preprocessing.food_image_processor import FoodImageProcessor
from src.preprocessing.menu_match_processor import MenuMatchPrecessor
from src.preprocessing.nutrition_processor import NutritionProcessor
from src.preprocessing.unimib_processor import UnimibProcessor

"""
The following script will read, resize, and save files from all of the experiment datasets.
The dataset consists of two file formats: .jpg and .h264
Note, only the first frame of the .h264 video is taken as the image.
All images are resized to 224x224 and saved in .jpg
"""
tf.get_logger().setLevel('INFO')
if __name__ == "__main__":
    """
    Runtime Variables
    """

    SHOW_ERRORS = True
    THROW_ERROR = False
    settings = ProcessingSettings(SHOW_ERRORS, THROW_ERROR)
    """
    Processing 
    """
    processors = {
        "nutrition": NutritionProcessor(),
        "ecust": EcustfdProcessor(),
        "menu_match": MenuMatchPrecessor(),
        "unimi": UnimibProcessor(),
        "food_image": FoodImageProcessor()
    }

    for processor in processors.values():
        processor.process(settings)
