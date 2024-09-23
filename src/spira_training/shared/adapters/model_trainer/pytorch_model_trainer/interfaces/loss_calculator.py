from abc import ABC, abstractmethod
from typing import Sequence

from src.spira_training.shared.core.models.loss import Loss

from src.spira_training.shared.core.models.dataset import Label


class LossCalculator(ABC):
    @abstractmethod
    def calculate_loss(
        self, predictions: Sequence[Label], labels: Sequence[Label]
    ) -> Loss: ...

    @abstractmethod
    def recalculate_weights(self): ...
