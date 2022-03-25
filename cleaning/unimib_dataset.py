import pandas as pd

from cleaning.dataset import Dataset


class UnimibDataset(Dataset):
    id_col = "image_name"
    label_col = "class"

    def __init__(self):
        super().__init__("unimib2016", "annotations.xlsx")
        self._food_info: pd.DataFrame = pd.DataFrame()
        self.load_data()

    def get_label(self, image_name: str) -> str:
        """
        gets class corresponding to image
        :param image_name: name of the image
        :return: the food class
        """
        food_name = self.get_food_info().loc[image_name][self.label_col]
        return self.food2index.to_ingredients_tensor([food_name])

    def get_food_info(self) -> pd.DataFrame:
        """
        gets all the food info from the label file
        :return: the dataframe of food info
        """
        return self._food_info

    def load_data(self):
        self._food_info = pd.read_excel(self.label_file, index_col=self.id_col)
        for food_category in self._food_info[self.label_col].unique():
            self.food2index.add(food_category)
        self.food2index.save()
