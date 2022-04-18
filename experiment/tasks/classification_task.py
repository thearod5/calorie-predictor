import numpy as np
import tensorflow as tf

from cleaning.dataset import prepare_dataset, split_dataset
from cleaning.food_images_dataset import FoodImagesDataset
from cleaning.nutrition_dataset import Mode, NutritionDataset
from cleaning.unimib_dataset import UnimibDataset
from constants import CLASSIFICATION_SUBSET, N_EPOCHS, TEST_SPLIT_SIZE
from experiment.Food2Index import Food2Index
from experiment.tasks.base_task import ClassificationTask


class FoodClassificationTask(ClassificationTask):
    def __init__(self, base_model, n_epochs=N_EPOCHS, dataset_indices: [int] = CLASSIFICATION_SUBSET):
        super().__init__(base_model, n_outputs=len(Food2Index()), n_epochs=n_epochs)

        datasets = [UnimibDataset(),
                    FoodImagesDataset(),
                    NutritionDataset(mode=Mode.INGREDIENTS)]
        datasets = [d for i, d in enumerate(datasets) if i in dataset_indices]
        dataset = datasets[0].get_dataset(shuffle=False)
        image_count = len(datasets[0].get_image_paths())
        for d in datasets[1:]:
            dataset = dataset.concatenate(d.get_dataset(shuffle=False))
            image_count += len(d.get_image_paths())
        print("Datasets:", ", ".join([d.__class__.__name__ for d in datasets]))
        print("Image Count:", image_count)
        d_splits = list(map(prepare_dataset, split_dataset(dataset, image_count, TEST_SPLIT_SIZE)))
        train, validation = d_splits[0], d_splits[1]
        self._train = train
        self._validation = validation
        self._test = None
        print("Class Count:", self.get_food_count())

    def get_food_count(self):
        count = 0
        for dataset in [self._train, self._test, self._validation]:
            if dataset is None:
                continue
            count += len(get_food_counts(dataset))
        return count

    def get_training_data(self):
        return self._train

    def get_validation_data(self):
        return self._validation

    def get_test_data(self):
        return self._test


def get_food_counts(dataset: tf.data.Dataset):
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
