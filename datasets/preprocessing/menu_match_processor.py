import os

from datasets.preprocessing.abstract_processor import AbstractProcessor, ProcessingPaths
from datasets.preprocessing.runner import PATH_TO_OUTPUT_DIR, PATH_TO_PROJECT, IMAGES_DIR
from datasets.menu_match_dataset import MenuMatchDataset


class MenuMatchPrecessor(AbstractProcessor):
    PATH_TO_MENU_MATCH = os.path.join(PATH_TO_PROJECT, "menu_match_dataset")
    PATH_TO_MENU_MATCH_INPUT = os.path.join(PATH_TO_MENU_MATCH, "foodimages")
    PATH_TO_MENU_MATCH_OUTPUT = os.path.join(PATH_TO_OUTPUT_DIR, MenuMatchDataset.DIR_NAME, IMAGES_DIR)
    ITEMS_INFO_FILE = os.path.join(PATH_TO_MENU_MATCH, 'items_info.txt')
    LABEL_FILE = os.path.join(PATH_TO_MENU_MATCH, 'labels.txt')
    DATA_FILE = os.path.join(PATH_TO_MENU_MATCH, MenuMatchDataset.DATA_FILE)

    MISSING_IMGS = {'img10.jpg'}

    def __init__(self):
        super().__init__(self.PATH_TO_MENU_MATCH_INPUT)

    def pre_process(self):
        if not os.path.exists(self.DATA_FILE):
            image_ingredient_mappings = self.get_image_ingredient_mappings()
            # TODO

    def create_output_paths(self, entry_name: str) -> ProcessingPaths:
        return self.create_generic_single_output(entry_name, self.PATH_TO_MENU_MATCH_OUTPUT)

    def convert_to_csv(self):
        new_data_file = self._get_new_filepath(self.ORIG_DATA_FILE, ".csv")
        with open(new_data_file, "w") as new_file:
            with open(self.ORIG_DATA_FILE, "r") as orig_file:
                for line in orig_file.readlines():
                    new_line = "".join(line.split())
                    new_line = new_line.replace('\t', '').replace(';', ',')
                    new_file.write(new_line + '\n')

    def _get_image_ingredient_mappings(self):
        image_ingredient_mappings = {}
        with open(self.ORIG_LABEL_FILE) as f:
            for line in f.readlines():
                img_data = "".join(line.split()).split(";")
                image_ingredient_mappings[img_data[0]] = img_data[1:-1]
        return image_ingredient_mappings

    def make_image_calorie_mapping_yml(self):
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

    def _remove_missing_imgs(self):
        for missing_img in self.MISSING_IMGS:
            self.image_ingredient_mappings.pop(missing_img)  # this image is missing??

    @staticmethod
    def _get_new_filepath(orig_filepath, new_ext):
        base_path, ext = os.path.splitext(orig_filepath)
        new_data_file = base_path + new_ext
        return new_data_file
