import os
from cleaning.volume_dataset import VolumeDataset
from constants import DATA_DIR


class CalorieDataset(VolumeDataset):
    calorie_dir = os.path.join(DATA_DIR, 'calorie')

    def __init__(self):
        super().__init__(follow_links=True)
        self.create_symlink()

    def create_symlink(self):
        sym_link = os.path.join(self.volume_dir, 'calorie_images')
        if not os.path.exists(sym_link):
            os.symlink(self.calorie_dir, sym_link)
