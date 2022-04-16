from cleaning.menu_match_dataset import MenuMatchDataset
from cleaning.nutrition_dataset import Mode, NutritionDataset
from constants import N_EPOCHS, TEST_SPLIT_SIZE
from experiment.tasks.base_task import RegressionTask, TaskType


class CaloriePredictionTask(RegressionTask):
    def __init__(self, base_model, n_epochs=N_EPOCHS):
        super().__init__(base_model, n_epochs=n_epochs)
        dataset = NutritionDataset(Mode.CALORIE)
        test_dataset = MenuMatchDataset()
        train, validation = dataset.split_to_train_test(TEST_SPLIT_SIZE)
        self._train = train
        self._validation = validation
        self._test = test_dataset.split_to_train_test().pop()

    def get_training_data(self):
        return self._train

    def get_validation_data(self):
        return self._validation

    def get_test_data(self):
        return self._test
