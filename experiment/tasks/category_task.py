from cleaning.nutrition_dataset import Mode, NutritionDataset
from constants import N_EPOCHS, TEST_SPLIT_SIZE
from experiment.IngredientIndexMap import IngredientIndexMap
from experiment.tasks.base_task import Task, TaskType


class FoodClassificationTask(Task):
    def __init__(self, base_model, n_epochs=N_EPOCHS):
        dataset = NutritionDataset(Mode.INGREDIENTS)
        ingredient_index = IngredientIndexMap()
        n_classes = len(ingredient_index)
        super().__init__(base_model, TaskType.CLASSIFICATION, n_outputs=n_classes, n_epochs=n_epochs)

        train, validation = dataset.split_to_train_test(TEST_SPLIT_SIZE)
        self._train = train
        self._validation = validation
        self._test = dataset.get_images()  # MenuMatchDataset()

    def training_data(self):
        return self._train

    def validation_data(self):
        return self._validation

    def test_data(self):
        return self._test
