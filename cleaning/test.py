from cleaning.calorie_dataset import CalorieDataset
from cleaning.volume_dataset import VolumeDataset
import unittest
from constants import *
from math import *


class TestDatasets(unittest.TestCase):
    def test_calorie_dataset(self):
        total_images = 3624
        calorie = CalorieDataset()
        self.perform_test(calorie, total_images)

    def itest_volume_dataset(self):
        total_images = 2978
        volume = VolumeDataset()
        self.perform_test(volume, total_images)

    def perform_test(self, dataset, total_images):
        validation_data = dataset.get_testing_data()
        training_data = dataset.get_training_data()

        validation_batches = list(validation_data.as_numpy_iterator())
        training_batches = list(training_data.as_numpy_iterator())
        validation_size = total_images * VALIDATION_SPLIT
        training_size = total_images - validation_size
        print(training_size / BATCH_SIZE)
        for val in validation_batches:
            print(val)
            break
        self.assertEqual(len(validation_batches), ceil(validation_size / BATCH_SIZE))
        self.assertEqual(len(training_batches), ceil(training_size / BATCH_SIZE))



if __name__ == '__main__':
    unittest.main()
