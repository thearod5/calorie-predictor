import os
from cleaning.dataset import Dataset
from constants import DATA_DIR
import csv
from ingredient import Ingredient
import pandas as pd


class VolumeDataset(Dataset):
    volume_dir = os.path.join(DATA_DIR, 'volume')
    nutrition_dir = os.path.join(volume_dir, 'nutrition5k')
    eucstfd_dir = os.path.join(volume_dir, 'eucstfd')
    name_index = 0
    calorie_index = 1
    mass_index = 2
    num_features = 6
    metadata_file1 = os.path.join(nutrition_dir, 'dish_metadata_cafe2.csv')
    density_file = os.path.join(eucstfd_dir, 'density.xls')
    image_ingredient_mappings = None
    label_to_remove = "ingr_"
    food_types = ['apple', 'banana', 'bread', 'bun', 'doughnut', 'egg', 'fired_dough_twist', 'grape', 'lemon', 'litchi', 'mango',
                  'mix', 'mooncake', 'orange', 'pear', 'peach', 'plum', 'qiwi', 'sachima', 'tomato']

    def __init__(self, follow_links=False):
        super().__init__(self.volume_dir, follow_links=follow_links)

    def get_image_ingredient_mappings(self):
        if self.image_ingredient_mappings is None:
            image_ingredient_mappings = {}
            with open(self.metadata_file1, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    new_row = [item for item in row if self.label_to_remove not in item]
                    ingredients = []
                    num_ingredients = int((len(new_row) / self.num_features) - 1)
                    for i in range(1, num_ingredients):
                        index = i * self.num_features
                        name = row[index + self.name_index]
                        calories = row[index + self.calorie_index]
                        mass = row[index + self.mass_index]
                        ingredients.append(Ingredient(name, calories, mass))
                    image_ingredient_mappings[row[self.name_index]] = ingredients
        return self.image_ingredient_mappings

    def get_volume_info(self):
        dataframes = pd.read_excel(self.density_file, sheet_name=self.food_types, index_col=False)
        df = pd.concat(dataframes.values(), ignore_index=True)
        return df

    def get_calories_for_label(self, id_):
        try:
            calories = self.get_volume_info().loc[id_]['weight(g)']
        except KeyError:
            calories = 0
        return calories


if __name__ == "__main__":
    print(VolumeDataset().something())
