from constants import N_EPOCHS
from experiment.tasks.calories_task import CaloriePredictionTask


class CalorieTransferTask(CaloriePredictionTask):

    def __init__(self, *args, **kwargs):
        super().__init__("resnet", n_epochs=N_EPOCHS, model_task="MassPredictionTask")

    def get_training_data(self):
        return self._train

    def get_validation_data(self):
        return self._validation

    def get_test_data(self):
        return self._test

    def get_eval_dataset(self, name: str) -> [str]:
        pass  # TODO
