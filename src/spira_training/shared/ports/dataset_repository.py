from abc import ABC, abstractmethod
from pathlib import Path
from src.spira_training.shared.core.models.dataset import Dataset


class DatasetRepository(ABC):
    @abstractmethod
    async def get_dataset(self, path: Path) -> Dataset:
        pass

    @abstractmethod
    async def save_dataset(self, dataset: Dataset, path: Path) -> None:
        pass
