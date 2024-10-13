from typing import List, Sequence

from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.dataloader import (
    Dataloader,
)
from src.spira_training.shared.core.models.batch import Batch
from tests.fakes.fake_dataset_repository import make_dataset


class FakeDataloader(Dataloader):
    def __init__(self, batches: List[Batch]):
        self._batches = batches

    def get_batches(self) -> Sequence[Batch]:
        return self._batches


def make_batch():
    return make_dataset()


def make_batches(
    length: int = 3,
):
    return [make_batch() for _ in range(length)]


def make_dataloader(
    batches: List[Batch] | None = None,
):
    return FakeDataloader(batches=batches or make_batches())
