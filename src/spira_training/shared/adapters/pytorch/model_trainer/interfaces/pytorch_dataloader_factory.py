from abc import ABC, abstractmethod

from .pytorch_dataloader import PytorchDataloader
from src.spira_training.shared.core.models.dataset import Dataset


class PytorchDataloaderFactory(ABC):
    @abstractmethod
    def make_dataloader(self, dataset: Dataset) -> PytorchDataloader: ...
