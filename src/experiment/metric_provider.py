from typing import List


class MetricProvider:
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """
        Returns the average absolute error between each y value.
        :param y_true: List of true values.
        :param y_pred: List of predicted values.
        :return: The average error.
        """
        errors = []
        for _y_true, _y_pred in zip(y_true, y_pred):
            error = abs(_y_true - _y_pred)
            errors.append(error)
        return sum(errors) / len(y_true)

    @staticmethod
    def error_of_average(y_true: List[float], y_pred: List[float]):
        """
        Returns the absolute different between the average of y_true and y_pred.
        :param y_true: List of true values.
        :param y_pred: List of predicted values.
        :return: The absolute value of the difference between means.
        """
        y_true_mean = sum(y_true) / len(y_true)
        y_pred_mean = sum(y_pred) / len(y_pred)
        return abs(y_true_mean - y_pred_mean)
