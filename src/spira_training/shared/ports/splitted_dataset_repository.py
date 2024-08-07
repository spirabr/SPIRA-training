from abc import ABC, abstractmethod

from src.spira_training.shared.models.splitted_dataset import SplittedDataset


class SplittedDatasetRepository(ABC):
    @abstractmethod
    def get_splitted_dataset(self, path: str) -> SplittedDataset:
        pass

    @abstractmethod
    def save_splitted_dataset(self, splitted_dataset: SplittedDataset, path: str) -> None:
        pass