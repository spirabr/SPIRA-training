from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.dataset import Dataset
from src.spira_training.shared.core.models.splitted_dataset import SplittedDataset


class DatasetSplitter(ABC):
    @abstractmethod
    def split(self, dataset: Dataset) -> SplittedDataset:
        pass
