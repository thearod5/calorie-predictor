from typing import List

import pandas as pd

from datasets.abstract_dataset import AbstractDataset, DatasetPathCreator


class EucstfdDataset(AbstractDataset):
    FOOD_TYPES = ['apple', 'banana', 'bread', 'bun', 'doughnut', 'egg', 'fired_dough_twist', 'grape', 'lemon', 'litchi',
                  'mango',
                  'mix', 'mooncake', 'orange', 'pear', 'peach', 'plum', 'qiwi', 'sachima', 'tomato']
    LABEL_COL = 'weight(g)'
    ID_COL = 'id'
    dataset_paths_creator = DatasetPathCreator(dataset_dirname='ecustfd', label_filename='density.xls')

    def __init__(self, use_ingredients_mass=False):
        self._food_info = None
        self.use_ingredients_mass = use_ingredients_mass
        super().__init__(self.dataset_paths_creator)

    def get_food_info(self) -> pd.DataFrame:
        """
        gets all the food info from the label file
        :return: the dataframe of food info
        """
        if self._food_info is None:
            excel_sheets = pd.read_excel(self.label_file, sheet_name=self.FOOD_TYPES, index_col=self.ID_COL)
            self._food_info = pd.concat(excel_sheets.values(), ignore_index=False)
        return self._food_info

    def get_label(self, image_name: str) -> List[float]:
        """
        gets the weights corresponding to image
        :param image_name: name of the image
        :return: a list of weights for the image
        """
        image_name = image_name.split('(')[0][:-1]  # TODO probably a better way to do this but idk
        labels = self.get_food_info().loc[image_name][self.LABEL_COL]
        if self.use_ingredients_mass:
            labels = list(labels) if isinstance(labels, pd.Series) else [labels]
        elif isinstance(labels, pd.Series):
            labels = sum(labels)
        return labels
