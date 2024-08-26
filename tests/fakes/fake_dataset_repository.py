from src.spira_training.shared.core.models.dataset import Dataset
from src.spira_training.shared.ports.dataset_repository import DatasetRepository


class FakeDatasetRepository(DatasetRepository):
    def __init__(self):
        self._datasets = {}

    async def get_dataset(self, path: str) -> Dataset:
        return self._datasets[path]

    async def save_dataset(self, dataset: Dataset, path: str) -> None:
        self._datasets[path] = dataset


def make_dataset():
    return Dataset(
        features=[],
        labels=[],
    )
