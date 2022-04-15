from cleaning.dataset import prepare_dataset, split_dataset
from cleaning.food_images_dataset import FoodImagesDataset
from cleaning.unimib_dataset import UnimibDataset
from constants import N_EPOCHS, TEST_SPLIT_SIZE
from experiment.Food2Index import Food2Index
from experiment.tasks.base_task import ClassificationTask, TaskType


class FoodClassificationTask(ClassificationTask):
    def __init__(self, base_model, n_epochs=N_EPOCHS):
        super().__init__(base_model, TaskType.CLASSIFICATION, n_outputs=len(Food2Index()), n_epochs=n_epochs)

        datasets = [FoodImagesDataset(), UnimibDataset()]
        dataset = datasets[0].get_dataset(shuffle=False)
        image_count = len(datasets[0].get_image_paths())
        for d in datasets[1:]:
            dataset = dataset.concatenate(d.get_dataset(shuffle=False))
            image_count += len(d.get_image_paths())
        # TODO: Uncomment after testing
        # dataset = dataset.shuffle(buffer_size=MAXIMUM_BUFFER_SIZE, seed=RANDOM_SEED)
        d_splits = list(map(prepare_dataset, split_dataset(dataset, image_count, TEST_SPLIT_SIZE)))
        train, validation = d_splits[0], d_splits[1]
        self._train = train
        self._validation = validation
        self._test = None

    def training_data(self):
        return self._train

    def validation_data(self):
        return self._validation

    def test_data(self):
        return self._test
