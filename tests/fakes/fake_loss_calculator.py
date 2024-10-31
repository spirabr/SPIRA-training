from typing import Sequence

from src.spira_training.shared.adapters.pytorch.models.pytorch_label import PytorchLabel
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_loss_calculator import (
    Loss,
    PytorchLossCalculator,
)


class FakeLossCalculator(PytorchLossCalculator):
    def __init__(self):
        self.loss = make_loss()
        self.recalculate_weights_calls = 0

    def calculate_loss(
        self, predictions: Sequence[PytorchLabel], labels: Sequence[PytorchLabel]
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


class FakeCyclingLossCalculator(FakeLossCalculator):
    def __init__(self):
        self.cycling_losses = []
        self.recalculate_weights_calls = 0
        self._calculate_loss_calls = 0

    def calculate_loss(
        self, predictions: Sequence[PytorchLabel], labels: Sequence[PytorchLabel]
    ) -> Loss:
        loss = self.cycling_losses[
            self._calculate_loss_calls % len(self.cycling_losses)
        ]
        self._calculate_loss_calls += 1
        return loss

    def recalculate_weights(self):
        self.recalculate_weights_calls += 1

    def with_fixed_losses(self, losses: Sequence[Loss]):
        self.cycling_losses = losses
        self._calculate_loss_calls = 0
        return self

    def assert_recalculate_weights_was_called(self, times: int):
        assert (
            self.recalculate_weights_calls == times
        ), f"Expected {times} calls, got {self.recalculate_weights_calls}"


def make_loss() -> Loss:
    return Loss(
        value=0.0,
    )
