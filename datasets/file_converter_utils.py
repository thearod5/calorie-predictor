import csv
import shutil

import pandas as pd
import yaml

from constants import *

"""
MENU MATCH
"""

MENU_MATCH = os.path.join(get_data_dir(), 'menu_match')

NEW_LABEL_FILE = os.path.join(MENU_MATCH, 'labels.yml')

NEW_DATA_FILE = os.path.join(MENU_MATCH, 'items_info.csv')
TOTAL_CALORIES_FILE = os.path.join(MENU_MATCH, 'total_calories.yml')







"""
NUTRITION5k
"""
NUTRITION_5k = os.path.join(get_data_dir(), 'nutrition5k')
ORIG_METADATA_FILE = os.path.join(NUTRITION_5k, 'dish_metadata_cafe2.csv')
NEW_METADATA_FILE = os.path.join(NUTRITION_5k, 'dish_metadata_cafe2.csv')


def make_new_metadata_csv():
    label_to_remove = "ingr_"
    with open(NEW_METADATA_FILE, "w") as new_file:
        writer = csv.writer(new_file)
        with open(ORIG_METADATA_FILE, newline='') as orig_file:
            reader = csv.reader(orig_file)
            for row in reader:
                new_row = [item for item in row if label_to_remove not in item]
                writer.writerow(new_row)


"""
Food Images
"""
FOOD_IMAGES_DIR = os.path.join(get_data_dir(), 'food_images')
FOOD_LABEL_FILE = os.path.join(FOOD_IMAGES_DIR, 'labels.yml')
IMAGES_DIR = os.path.join(FOOD_IMAGES_DIR, 'images')


def move_images_and_generate_yml(generate_yml=True):
    image_class_mapping = {}
    for root, dirs, files in os.walk(FOOD_IMAGES_DIR):
        if root == FOOD_IMAGES_DIR or root == IMAGES_DIR:
            continue
        class_ = root.split(os.sep)[-1]
        for filename in files:
            name = filename.split(EXT_SEP)[0]
            image_class_mapping[name] = class_
            shutil.move(os.path.join(root, filename), IMAGES_DIR)

    if generate_yml:
        with open(FOOD_LABEL_FILE, 'w') as file:
            yaml.dump(image_class_mapping, file)
