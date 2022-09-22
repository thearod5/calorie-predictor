import os

import pandas as pd

from cleaning.file_converter_utils import MENU_MATCH

ROOT = "https://menu-match.s3.us-east-2.amazonaws.com/"
DATA_DIR = os.path.join(MENU_MATCH, "images")

if __name__ == "__main__":
    image_names = []
    for image_file in os.listdir(DATA_DIR):
        image_path = os.path.join(ROOT, image_file)
        image_names.append(image_path)
    image_names.sort()

    # Step - Create csv
    data = pd.DataFrame()
    data["image_url"] = image_names
    data.to_csv("images.csv", index=False)
