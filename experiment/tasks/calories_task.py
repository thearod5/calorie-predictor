from datasets.menu_match_dataset import MenuMatchDataset
from datasets.nutrition_dataset import Mode, NutritionDataset
from constants import N_EPOCHS, TEST_SPLIT_SIZE
from experiment.tasks.base_task import RegressionTask


class CaloriePredictionTask(RegressionTask):

    def __init__(self, base_model, n_epochs=N_EPOCHS):
        super().__init__(base_model, n_epochs=n_epochs)
        dataset = NutritionDataset(Mode.CALORIE)
        train, validation = dataset.split_to_train_test(TEST_SPLIT_SIZE)
        test_dataset = MenuMatchDataset()
        self._train = train
        self._validation = validation
        self._test = test_dataset.split_to_train_test().pop()

    def get_training_data(self):
        return self._train

    def get_validation_data(self):
        return self._validation

    def get_test_data(self):
        return self._test

    def get_eval_dataset(self, name: str) -> [str]:
        pass  # TODO
