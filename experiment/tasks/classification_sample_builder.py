from constants import CLASSIFICATION_SUBSET
from experiment.tasks.base_task import BaseModel
from experiment.tasks.classification_task import FoodClassificationTask


class ClassificationSubsetTask(FoodClassificationTask):
    def __init__(self, base_model: BaseModel, n_epochs):
        super().__init__(base_model=base_model, n_epochs=n_epochs, dataset_indices=CLASSIFICATION_SUBSET)
