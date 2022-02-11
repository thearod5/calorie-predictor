from constants import *
import tensorflow as tf


class Dataset:

    def __init__(self, dataset_dir, labels=None, follow_links=False, label_mode=None):
        self.dataset_dir = dataset_dir
        self.labels = labels
        self.label_mode = label_mode
        self.follow_links = follow_links
        self._training_data = None
        self._validation_data = None

    def get_params(self, subset="training"):
        return {'directory': self.dataset_dir, 'validation_split': VALIDATION_SPLIT, 'subset': subset,
                'color_mode': 'rgb', 'image_size': IMAGE_SIZE, 'label_mode': self.label_mode, 'labels': self.labels,
                'shuffle': True, 'seed': RANDOM_SEED, 'batch_size': BATCH_SIZE, 'follow_links': self.follow_links}

    def get_training_data(self):
        if self._training_data is None:
            self._training_data = tf.keras.utils.image_dataset_from_directory(
                **self.get_params("training")
            )
        return self._training_data

    def get_testing_data(self):
        if self._validation_data is None:
            self._validation_data = tf.keras.utils.image_dataset_from_directory(
                **self.get_params("validation")
            )
        return self._validation_data
