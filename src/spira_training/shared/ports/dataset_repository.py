from abc import ABC, abstractmethod
from typing import List

class DatasetRepository(ABC):
    @abstractmethod
    def get_dataset(self, path: str) -> Dataset:
        pass

    @abstractmethod
    def save_dataset(self, dataset: Dataset, path: str) -> None:
        pass