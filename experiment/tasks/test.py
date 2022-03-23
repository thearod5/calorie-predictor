from cleaning.menu_match_dataset import MenuMatchDataset
from constants import N_EPOCHS, TEST_SPLIT_SIZE
from experiment.tasks.task import Task, TaskType


class TestTask(Task):
    def __init__(self, base_model, n_epochs=N_EPOCHS):
        super().__init__(base_model, TaskType.REGRESSION, n_epochs=n_epochs)
        dataset = MenuMatchDataset()
        train, validation = dataset.split_to_train_test(TEST_SPLIT_SIZE)
        self._train = train
        self._validation = validation

    def training_data(self):
        return self._train

    def validation_data(self):
        return self._validation

    def test_data(self):
        # NO TEST DATA
        return self._validation
