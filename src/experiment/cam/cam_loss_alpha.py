from enum import Enum


class AlphaStrategy(Enum):
    CONSTANT = "constant"
    ADAPTIVE = "adaptive"


class CamLossAlpha:
    def __init__(self, total_epochs: int, alpha: float = None, alpha_decay: float = None):
        self.assert_valid_init(alpha, alpha_decay)
        self.alpha_strategy = AlphaStrategy.CONSTANT if alpha else AlphaStrategy.ADAPTIVE
        self.alpha = alpha if alpha else 1
        self.alpha_decay = alpha_decay
        self.n_alpha_steps = 1 / alpha_decay
        self.n_alpha_updates = int(total_epochs / self.n_alpha_steps)
        self.n_epochs = 1

    @staticmethod
    def assert_valid_init(alpha: float = None, alpha_decay: float = None) -> None:
        """
        Asserts that either alpha or alpha_decay are defined, but not both.
        :param alpha: The starting alpha value.
        :param alpha_decay: The step at which alpha should decay.
        :return: None
        """
        not_enough_error = "Expected one of (alpha_decay) or (alpha) to be defined."
        too_much_error = "Did not expect both (alpha_decay) or (alpha) to be defined."
        assert not (alpha is None and alpha_decay is None), not_enough_error
        assert alpha is not None and alpha_decay is not None, too_much_error

    def perform_alpha_step(self) -> bool:
        """
        Updates alpha according to defined alpha strategy.
        :return: Whether alpha was updated this step.
        """
        self.n_epochs += 1
        if self.alpha_strategy == AlphaStrategy.CONSTANT:
            return False
        if self.n_epochs % self.n_alpha_updates == 0:
            self.alpha -= self.alpha_decay
            self.alpha = round(self.alpha, 2)
            return True
        return False
