from abc import ABC, abstractmethod
from typing import Sequence

from src.spira_training.shared.core.models.loss import Loss
from src.spira_training.shared.adapters.pytorch.models.pytorch_label import (
    PytorchLabel,
)


class PytorchLossCalculator(ABC):
    @abstractmethod
    def calculate_loss(
        self, predictions: Sequence[PytorchLabel], labels: Sequence[PytorchLabel]
    ) -> Loss: ...

    @abstractmethod
    def recalculate_weights(self): ...
