from typing import Callable, Dict

import numpy as np
from tensorflow.python.data import Dataset

from constants import CLASSIFICATION_DATASETS, N_EPOCHS, TEST_SPLIT_SIZE
from datasets.abstract_dataset import AbstractDataset
from datasets.food_images_dataset import FoodImagesDataset
from datasets.nutrition_dataset import Mode, NutritionDataset
from datasets.unimib_dataset import UnimibDataset
from experiment.Food2Index import Food2Index
from experiment.tasks.base_task import ClassificationTask


class FoodClassificationTask(ClassificationTask):
    dataset_constructors: Dict[str, Callable[[], AbstractDataset]] = {
        "unimib": lambda: UnimibDataset(),
        "food_images": lambda: FoodImagesDataset(),
        "nutrition5k": lambda: NutritionDataset(mode=Mode.INGREDIENTS)
    }

    def __init__(self, base_model, n_epochs=N_EPOCHS, training_datasets: [str] = CLASSIFICATION_DATASETS):
        super().__init__(base_model, n_outputs=len(Food2Index()), n_epochs=n_epochs)
        datasets = self.get_datasets(training_datasets)
        dataset, image_count = self.combine_datasets(datasets)
        d_splits = list(map(Dataset.prepare_dataset, Dataset.split_dataset(dataset, image_count, TEST_SPLIT_SIZE)))
        train, validation = d_splits[0], d_splits[1]
        self._train = train
        self._validation = validation
        self._test = None

    def get_eval_dataset(self, name) -> [str]:
        if name not in self.dataset_constructors:
            raise Exception("%s is not one of %s" % (name, ", ".join(self.dataset_constructors.keys())))
        return Dataset.prepare_dataset(self.dataset_constructors[name]().get_dataset(shuffle=False))

    def get_datasets(self, dataset_names) -> [AbstractDataset]:
        return [self.dataset_constructors[d_name]() for d_name in self.dataset_constructors if d_name in dataset_names]

    def get_food_count(self):
        count = 0
        for dataset in [self._train, self._test, self._validation]:
            if dataset is None:
                continue
            count += len(self.get_food_counts(dataset))
        return count

    def get_training_data(self):
        return self._train

    def get_validation_data(self):
        return self._validation

    def get_test_data(self):
        return self._test

    @staticmethod
    def get_food_counts(dataset: AbstractDataset):
        food2index = Food2Index()
        food2count = {}
        for batch_images, batch_labels in dataset:
            for batch_index in range(batch_images.shape[0]):
                label = batch_labels[batch_index]
                label_indices = np.where(label == 1)[0]
                foods = []
                for label_index in label_indices:
                    food_name = food2index.get_ingredient(label_index)
                    foods.append(food_name)
                    if food_name in food2count:
                        food2count[food_name] += 1
                    else:
                        food2count[food_name] = 1

        return food2count
