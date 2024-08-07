from abc import ABC, abstractmethod
from typing import List
from src.spira_training.shared.models.dataset import Dataset


class DatasetRepository(ABC):
    @abstractmethod
    def get_dataset(self, path: str) -> Dataset:
        pass

    @abstractmethod
    def save_dataset(self, dataset: Dataset, path: str) -> None:
        pass
