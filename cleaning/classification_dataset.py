import os
from cleaning.dataset import Dataset
from constants import DATA_DIR


class ClassificationDataset(Dataset):
    classification_dir = os.path.join(DATA_DIR, 'classification')

    def __init__(self):
        labels = self.get_labels()
        super().__init__(self.classification_dir, labels=labels, label_mode='categorical')

    def get_labels(self):
        # TODO
        return []
