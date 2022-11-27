from typing import Dict, List

import tensorflow as tf

from src.experiment.cam.cam_logger import CamLogger


class CamState:
    VALIDATION_SCORE_PARAM = "Best Validation Score"
    CAL_LOSS_PARAM = "Total Calorie Loss"
    HMAP_LOSS_PARAM = "Total HMAP Loss"
    COMPOSITE_LOSS_PARAM = "Total Composite Loss"
    EPOCH_PARAM = "Epoch"
    STEP_PARAM = "Step"
    STATE_PARAM = "state"

    """
    Logs training stats for CamTrainer.
    """

    def __init__(self, log_dir: str, batch_size: int, load_previous: bool = False):
        """
        Instantiates logger as new run and with log path.
        :param log_dir: Path to store logs.
        """
        self.batch_size = batch_size
        self.epoch: int = 0
        self.composite_losses: List[float] = []
        self.composite_losses: List[float] = []
        self.validation_losses: List[float] = []
        self.calorie_losses: List[float] = []
        self.validation_accuracy: List[float] = []
        self.n_images: float = 0
        self.n_steps: int = 0
        self.total_calorie_loss: tf.Tensor = 0
        self.total_hmap_loss: tf.Tensor = 0
        self.total_composite_loss: tf.Tensor = 0
        self.validation_score = None
        self.train_loss = []
        self.logger = CamLogger(log_dir)

        if load_previous:
            self.logger.load()
            self.load_log(self.logger.logs[-1])

    def log_step(self, composite_loss: float, calorie_loss: float, feature_losss: float, alpha: float,
                 do_export: bool = True,
                 do_print: bool = True, **kwargs):
        """
        Records the composite and class losses in aggregate tallies.
        :param feature_losss: The loss of between the model's features and human maps.
        :param composite_loss: The combine class and hmap loss.
        :param calorie_loss: The loss of the calorie predictions.
        :param do_export: Whether to export current log.
        :param do_print: Whether to print current losses.
        :return: None
        """
        self.total_composite_loss += composite_loss
        self.total_calorie_loss += calorie_loss
        self.total_hmap_loss += feature_losss
        self.n_images += self.batch_size
        self.composite_losses.append(composite_loss)
        self.n_steps += 1
        average_loss = self.total_calorie_loss / self.n_steps
        if do_print:
            print("\rEpoch:", self.epoch,
                  "\tStep:", self.n_steps,
                  "\tCurrent Error:", calorie_loss.numpy(),
                  "\tAverage:", average_loss.numpy(),
                  "\tAlpha:", alpha)
        if do_export:
            self.export_log(**kwargs)

    def log_epoch(self, do_export=True) -> None:
        """
        Stores the average mse and loss for the epoch and exports it.
        :param do_export: Whether to export log after operation.
        :return: None
        """
        self.calorie_losses.append(self.total_calorie_loss)
        self.composite_losses.append(self.total_composite_loss)
        if do_export:
            self.export_log()
        self.epoch += 1
        self.reset_losses()

    def reset_losses(self):
        self.total_calorie_loss = 0
        self.total_hmap_loss = 0
        self.total_composite_loss = 0
        self.n_steps = 0

    def log_eval(self, score: float, do_export=True, metric_direction: str = "lower"):
        """
        Stores current validation score and returns whether it's the best one.
        :param score: The current score of the evaluation.
        :param do_export: Whether to export current logs.
        :param metric_direction: The direction in which the validation metric gets better.
        :return: Boolean representing whether score is current best.
        :rtype:
        """
        metric_direction = {
            "lower": lambda c, p: c < p,
            "greater": lambda c, p: c > p,
        }[metric_direction]

        if self.validation_score is None:
            self.validation_score = score
            is_better = True
        else:
            is_better = metric_direction(score, self.validation_score)

        if is_better:
            self.validation_score = score

        if do_export:
            self.export_log()
        return is_better

    def export_log(self, **kwargs) -> None:
        """
        Saves log in log path.
        :return: None
        """
        log = self.create_log()
        log.update(kwargs)
        self.logger.log(log)

    def get_average_loss(self, metric_name: str):
        """
        Returns the average loss for given variable.
        :param metric_name: The name of the metric whose loss is returned.
        :return:
        :rtype:
        """
        metric_value = {
            "calorie": self.total_calorie_loss,
            "hmap": self.total_hmap_loss,
            "composite": self.total_composite_loss
        }[metric_name]
        return float((metric_value / self.n_steps).numpy())

    def load_log(self, log: Dict):
        """
        Loads the log's state into instance.
        :param log: The log containing state variables to load.
        :return: None
        """
        state = log[self.STATE_PARAM]
        self.epoch = state[self.EPOCH_PARAM]
        self.n_steps = state[self.STATE_PARAM]
        self.validation_score = state[self.VALIDATION_SCORE_PARAM]
        self.total_calorie_loss = state[self.CAL_LOSS_PARAM]
        self.total_hmap_loss = state[self.HMAP_LOSS_PARAM]
        self.total_composite_loss = state[self.COMPOSITE_LOSS_PARAM]

    def create_log(self) -> Dict:
        """
        :return: Dictionary containing log information.
        """
        return {
            "metrics": {
                "Calorie Loss (avg)\t": self.get_average_loss("calorie"),
                "Hmap Loss (avg)\t": self.get_average_loss("hmap"),
                "Composite Loss (avg)\t": self.get_average_loss("composite")
            },
            self.STATE_PARAM: {
                self.EPOCH_PARAM: self.epoch,
                self.STEP_PARAM: self.n_steps,
                self.VALIDATION_SCORE_PARAM: self.validation_score,
                self.CAL_LOSS_PARAM: self.total_calorie_loss,
                self.HMAP_LOSS_PARAM: self.total_hmap_loss,
                self.COMPOSITE_LOSS_PARAM: self.total_composite_loss
            }
        }
