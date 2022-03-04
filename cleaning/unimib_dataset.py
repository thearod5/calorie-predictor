from cleaning.dataset import Dataset
import pandas as pd


class UnimibDataset(Dataset):
    id_col = "image_name"
    label_col = "class"

    def __init__(self):
        super().__init__("unimib2016", "annotations.xlsx")
        self._food_info = None

    def get_label(self, image_name: str) -> str:
        """
        gets class corresponding to image
        :param image_name: name of the image
        :return: the food class
        """
        return self.get_food_info().loc[image_name][self.label_col]

    def get_food_info(self) -> pd.DataFrame:
        """
        gets all the food info from the label file
        :return: the dataframe of food info
        """
        if self._food_info is None:
            self._food_info = pd.read_excel(self.label_file, index_col=self.id_col)
        return self._food_info
