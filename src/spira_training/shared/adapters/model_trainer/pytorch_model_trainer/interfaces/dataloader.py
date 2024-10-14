from abc import ABC, abstractmethod
from typing import Sequence

from spira_training.shared.adapters.model_trainer.pytorch_model_trainer.pytorch_batch import (
    PytorchBatch,
)


class Dataloader(ABC):
    @abstractmethod
    def get_batches(self) -> Sequence[PytorchBatch]: ...
