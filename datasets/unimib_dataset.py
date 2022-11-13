import pandas as pd

from datasets.abstract_dataset import AbstractDataset
from datasets.dataset_path_creator import DatasetPathCreator


class UnimibDataset(AbstractDataset):
    ID_COL = "image_name"
    LABEL_COL = "class"
    dataset_paths_creator = DatasetPathCreator(dataset_dir_name="unimib2016", label_filename="annotations.xlsx")

    def __init__(self):
        self._food_info: pd.DataFrame = pd.DataFrame()
        super().__init__(self.dataset_paths_creator)

    def get_label(self, image_name: str) -> str:
        """
        gets class corresponding to image
        :param image_name: name of the image
        :return: the food class
        """
        food_name = self.get_food_info().loc[image_name][self.LABEL_COL]
        return self.food2index.to_ingredients_tensor([food_name])

    def get_food_info(self) -> pd.DataFrame:
        """
        gets all the food info from the label file
        :return: the dataframe of food info
        """
        return self._food_info

    def load_data(self) -> None:
        """
        Loads the data for the dataset
        :return: None
        """
        self._food_info = pd.read_excel(self.label_file, index_col=self.ID_COL)
        for food_category in self._food_info[self.LABEL_COL].unique():
            self.food2index.add(food_category)
        self.food2index.save()
