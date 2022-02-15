import csv
import shutil

import yaml
import os
from constants import *
import pandas as pd

"""
MENU MATCH
"""

MENU_MATCH = os.path.join(DATA_DIR, 'menu_match')
ORIG_LABEL_FILE = os.path.join(MENU_MATCH, 'labels.txt')
NEW_LABEL_FILE = os.path.join(MENU_MATCH, 'labels.yml')
ORIG_DATA_FILE = os.path.join(MENU_MATCH, 'items_info.txt')
NEW_DATA_FILE = os.path.join(MENU_MATCH, 'items_info.csv')
TOTAL_CALORIES_FILE = os.path.join(MENU_MATCH, 'total_calories.yml')


def convert_to_csv():
    with open(NEW_DATA_FILE, "w") as new_file:
        with open(ORIG_DATA_FILE) as orig_file:
            for line in orig_file.readlines():
                new_line = "".join(line.split())
                new_line = new_line.replace('\t', '').replace(';', ',')
                new_file.write(new_line + '\n')


def make_image_ingredient_yml():
    missing_img = 'img10.jpg'
    image_ingredient_mappings = {}
    with open(ORIG_LABEL_FILE) as f:
        for line in f.readlines():
            img_data = "".join(line.split()).split(";")
            image_ingredient_mappings[img_data[0]] = img_data[1:-1]
    image_ingredient_mappings.pop(missing_img)  # this image is missing??
    with open(NEW_LABEL_FILE, 'w') as file:
        yaml.dump(image_ingredient_mappings, file)


def make_image_calorie_mapping_yml():
    food_info = pd.read_csv(NEW_DATA_FILE, index_col=1)
    image_labels_mapping = yaml.safe_load(open(NEW_LABEL_FILE))
    image_calorie_mappings = {}
    for img, labels in image_labels_mapping.items():
        total_calories = 0
        for label in labels:
            total_calories += food_info.loc[label]['Calories']
        image_calorie_mappings[img.split(".")[0]] = int(total_calories)
    with open(TOTAL_CALORIES_FILE, 'w') as file:
        yaml.dump(image_calorie_mappings, file)


"""
NUTRITION5k
"""
NUTRITION_5k = os.path.join(DATA_DIR, 'nutrition5k')
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
FOOD_IMAGES_DIR = os.path.join(DATA_DIR, 'food_images')
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
