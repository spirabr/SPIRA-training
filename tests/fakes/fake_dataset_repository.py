from pathlib import Path
import random
from src.spira_training.shared.core.models.dataset import Dataset, Label
from src.spira_training.shared.ports.dataset_repository import DatasetRepository
from tests.fakes.fake_audios_repository import make_audio


class FakeDatasetRepository(DatasetRepository):
    def __init__(self):
        self._datasets = {}
        self.save_dataset_called = False

    async def get_dataset(self, path: Path) -> Dataset:
        return self._datasets[path]

    async def save_dataset(self, dataset: Dataset, path: Path) -> None:
        self._datasets[path] = dataset
        self.save_dataset_called = True


def make_dataset(
    size: int = 100,
):
    features = [make_audio() for _ in range(0, size)]
    all_labels = [label for label in Label]
    features_labels = [random.choice(all_labels) for _ in features]
    return Dataset(features=features, labels=features_labels)
