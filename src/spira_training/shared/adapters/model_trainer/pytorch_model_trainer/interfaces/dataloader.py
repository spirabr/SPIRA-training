from abc import ABC, abstractmethod
from typing import Sequence

from src.spira_training.shared.core.models.batch import Batch


class Dataloader(ABC):
    @abstractmethod
    def get_batches(self) -> Sequence[Batch]: ...
