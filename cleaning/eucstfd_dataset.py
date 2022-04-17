from typing import List

from cleaning.dataset import Dataset
import pandas as pd


class EucstfdDataset(Dataset):
    food_types = ['apple', 'banana', 'bread', 'bun', 'doughnut', 'egg', 'fired_dough_twist', 'grape', 'lemon', 'litchi', 'mango',
                  'mix', 'mooncake', 'orange', 'pear', 'peach', 'plum', 'qiwi', 'sachima', 'tomato']
    label_col = 'weight(g)'
    id_col = 'id'
    split_characters = ['S', 'T']

    def __init__(self, use_ingredients_mass=False):
        """
        constructor
        """
        super().__init__('eucstfd', 'density.xls')
        self._food_info = None
        self.use_ingredients_mass = use_ingredients_mass

    def get_food_info(self) -> pd.DataFrame:
        """
        gets all the food info from the label file
        :return: the dataframe of food info
        """
        if self._food_info is None:
            excel_sheets = pd.read_excel(self.label_file, sheet_name=self.food_types, index_col=self.id_col)
            self._food_info = pd.concat(excel_sheets.values(), ignore_index=False)
        return self._food_info

    def get_label(self, image_name: str) -> List[float]:
        """
        gets the weights corresponding to image
        :param image_name: name of the image
        :return: a list of weights for the image
        """
        for char in self.split_characters:
            image_name = image_name.split(char)[0]  # TODO probably a better way to do this but idk
        labels = self.get_food_info().loc[image_name][self.label_col]
        if self.use_ingredients_mass:
            labels = list(labels) if isinstance(labels, pd.Series) else [labels]
        elif isinstance(labels, pd.Series):
            labels = sum(labels)
        return labels
