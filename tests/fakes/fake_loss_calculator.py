from typing import Sequence
from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.loss_calculator import (
    Loss,
    LossCalculator,
)
from src.spira_training.shared.core.models.dataset import Label


class FakeLossCalculator(LossCalculator):
    def __init__(self):
        self.loss = make_loss()

    def calculate_loss(
        self, predictions: Sequence[Label], labels: Sequence[Label]
    ) -> Loss:
        return self.loss

    def with_fixed_loss(self, loss: Loss):
        self.loss = loss
        return self


def make_loss() -> Loss:
    return Loss()
