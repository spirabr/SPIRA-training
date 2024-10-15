from abc import ABC, abstractmethod
from typing import Sequence

from src.spira_training.shared.adapters.pytorch.models.pytorch_batch import (
    PytorchBatch,
)


class Dataloader(ABC):
    @abstractmethod
    def get_batches(self) -> Sequence[PytorchBatch]: ...
