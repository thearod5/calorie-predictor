from enum import Enum


class AlphaStrategy(Enum):
    CONSTANT_BALANCED = "constant_balanced"
    ADAPTIVE = "adaptive"
    CONSTANT = "constant"


class CamLossAlpha:
    def __init__(self, total_epochs: int, alpha_strategy: AlphaStrategy, alpha_decay: float = 0.2):
        self.alpha_strategy = alpha_strategy
        self.n_epochs = 1
        if alpha_strategy == AlphaStrategy.CONSTANT_BALANCED:
            self.alpha = .5
            self.alpha_decay = None
        else:
            self.alpha = 1
            self.alpha_decay = alpha_decay
            self.n_alpha_steps = 1 / alpha_decay if alpha_decay else None
            self.n_alpha_updates = int(total_epochs / self.n_alpha_steps) if alpha_decay else None
        print("Alpha:", "\tStrategy:", self.alpha_strategy, "\talpha:", self.alpha, "\tdecay:", self.alpha_decay)

    def perform_alpha_step(self) -> bool:
        """
        Updates alpha according to defined alpha strategy.
        :return: Whether alpha was updated this step.
        """
        self.n_epochs += 1
        if self.alpha_strategy == AlphaStrategy.CONSTANT_BALANCED:
            return False
        if self.n_epochs % self.n_alpha_updates == 0:
            self.alpha -= self.alpha_decay
            self.alpha = round(self.alpha, 2)
            return True
        return False
