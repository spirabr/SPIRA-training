from typing import Sequence
from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.loss_calculator import (
    Loss,
    LossCalculator,
)
from src.spira_training.shared.core.models.dataset import Label


class FakeLossCalculator(LossCalculator):
    def __init__(self):
        self.loss = make_loss()
        self.recalculate_weights_calls = 0

    def calculate_loss(
        self, predictions: Sequence[Label], labels: Sequence[Label]
    ) -> Loss:
        return self.loss

    def recalculate_weights(self):
        self.recalculate_weights_calls += 1

    def with_fixed_loss(self, loss: Loss):
        self.loss = loss
        return self

    def assert_recalculate_weights_was_called(self, times: int):
        assert (
            self.recalculate_weights_calls == times
        ), f"Expected {times} calls, got {self.recalculate_weights_calls}"


def make_loss() -> Loss:
    return Loss()
