from datasets.preprocessing.base_processor import ProcessingSettings
from datasets.preprocessing.ecustfd_processor import EcustfdProcessor
from datasets.preprocessing.food_image_processor import FoodImageProcessor
from datasets.preprocessing.menu_match_processor import MenuMatchPrecessor
from datasets.preprocessing.nutrition_processor import NutritionProcessor
from datasets.preprocessing.unimib_processor import UnimibProcessor

"""
The following script will read, resize, and save files from all of the experiment datasets.
The dataset consists of two file formats: .jpg and .h264
Note, only the first frame of the .h264 video is taken as the image.
All images are resized to 224x224 and saved in .jpg
"""

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
    NutritionProcessor().process(settings)
    EcustfdProcessor().process(settings)
    MenuMatchPrecessor().process(settings)
    UnimibProcessor().process(settings)
    FoodImageProcessor().process(settings)
