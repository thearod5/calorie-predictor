import os
from cleaning.dataset import Dataset
from constants import DATA_DIR


class VolumeDataset(Dataset):
    volume_dir = os.path.join(DATA_DIR, 'volume')

    def __init__(self, follow_links=False):
        super().__init__(self.volume_dir, follow_links=follow_links)
