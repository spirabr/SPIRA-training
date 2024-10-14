from abc import ABC, abstractmethod
from typing import Sequence

from src.spira_training.shared.core.models.loss import Loss
from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.pytorch_model import (
    PytorchLabel,
)


class LossCalculator(ABC):
    @abstractmethod
    def calculate_loss(
        self, predictions: Sequence[PytorchLabel], labels: Sequence[PytorchLabel]
    ) -> Loss: ...

    @abstractmethod
    def recalculate_weights(self): ...
