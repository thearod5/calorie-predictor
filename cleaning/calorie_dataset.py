import os
from cleaning.dataset import Dataset
from constants import DATA_DIR
import pandas


class CalorieDataset(Dataset):
    calorie_dir = os.path.join(DATA_DIR, 'calorie')
    image_dir = os.path.join(calorie_dir, 'images')
    menu_match_dir = os.path.join(calorie_dir, 'menu_match')
    labels_file = os.path.join(menu_match_dir, 'labels.txt')
    items_info_file = os.path.join(menu_match_dir, 'items_info.txt')
    items_info_csv = os.path.join(menu_match_dir, 'items_info.csv')
    image_labels_mapping = None
    image_calorie_mapping = None
    calorie_info = None
    calorie_labels = None
    volume_dir = os.path.join(DATA_DIR, 'volume')

    def __init__(self):
        super().__init__(follow_links=True, dataset_dir=self.calorie_dir, labels=self.get_calorie_labels(), label_mode='int')

    @staticmethod
    def convert_to_csv(self, orig_file_path, new_file_path):
        with open(new_file_path, "w") as new_file:
            with open(orig_file_path) as orig_file:
                for line in orig_file.readlines():
                    new_line = "".join(line.split())
                    new_line = new_line.replace('\t', '').replace(';', ',')
                    new_file.write(new_line + '\n')

    def get_image_labels_mapping(self):
        if self.image_labels_mapping is None:
            self.image_labels_mapping = {}
            with open(self.labels_file) as f:
                for line in f.readlines():
                    img_data = "".join(line.split()).split(";")
                    self.image_labels_mapping[img_data[0]] = img_data[1:-1]
            self.image_labels_mapping.pop('img10.jpg')  # TODO figure out why this image is missing
        return self.image_labels_mapping

    def get_image_calorie_mapping(self):
        if self.image_labels_mapping is None:
            image_labels_mapping = self.get_image_labels_mapping()
            self.image_calorie_mapping = {}
            for img, labels in image_labels_mapping.items():
                total_calories = 0
                for label in labels:
                    total_calories += self.get_calories_for_label(label)
                self.image_calorie_mapping[img] = total_calories
        return self.image_calorie_mapping

    def get_calorie_labels(self):
        if self.calorie_labels is None:
            image_labels_mapping = self.get_image_labels_mapping()
            self.calorie_labels = []
            for img, labels in image_labels_mapping.items():
                total_calories = 0
                for label in labels:
                    total_calories += self.get_calories_for_label(label)
                self.calorie_labels.append(total_calories)
        return self.calorie_labels

    def get_calorie_info(self):
        if self.calorie_info is None:
            if not os.path.exists(self.items_info_csv):
                CalorieDataset.convert_to_csv(self.items_info_file, self.items_info_csv)
            self.calorie_info = pandas.read_csv(self.items_info_csv, index_col=1)
        return self.calorie_info

    def get_calories_for_label(self, label):
        try:
            calories = self.get_calorie_info().loc[label]['Calories']
        except KeyError:
            calories = 0
        return calories
