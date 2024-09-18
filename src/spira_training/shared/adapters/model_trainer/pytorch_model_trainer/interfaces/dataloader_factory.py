from abc import ABC, abstractmethod

from .dataloader import Dataloader
from src.spira_training.shared.core.models.dataset import Dataset


class DataloaderFactory(ABC):
    @abstractmethod
    def make_dataloader(self, dataset: Dataset) -> Dataloader: ...
