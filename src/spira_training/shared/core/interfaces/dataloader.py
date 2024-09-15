from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.batch import Batch


class Dataloader(ABC):
    @abstractmethod
    def get_batch(self) -> Batch: ...
