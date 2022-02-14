from typing import List

from cleaning.dataset import Dataset
import pandas as pd


class EucstfdDataset(Dataset):
    food_types = ['apple', 'banana', 'bread', 'bun', 'doughnut', 'egg', 'fired_dough_twist', 'grape', 'lemon', 'litchi', 'mango',
                  'mix', 'mooncake', 'orange', 'pear', 'peach', 'plum', 'qiwi', 'sachima', 'tomato']
    label_col = 'weight(g)'
    id_col = 'id'
    split_characters = ['S', 'T']

    def __init__(self):
        super().__init__('eucstfd', 'density.xls')
        self._food_info = None

    def get_food_info(self) -> pd.DataFrame:
        if self._food_info is None:
            excel_sheets = pd.read_excel(self.label_file, sheet_name=self.food_types, index_col=self.id_col)
            self._food_info = pd.concat(excel_sheets.values(), ignore_index=False)
        return self._food_info

    def get_label(self, image_name: str) -> List[float]:
        for char in self.split_characters:
            image_name = image_name.split(char)[0]  # TODO probably a better way to do this but idk
        labels = self.get_food_info().loc[image_name][self.label_col]
        return [label for label in labels] if isinstance(labels, pd.Series) else [labels]
