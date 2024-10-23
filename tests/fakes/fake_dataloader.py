from typing import List, Sequence

from src.spira_training.shared.adapters.pytorch.models.pytorch_batch import (
    PytorchBatch,
)

from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.dataloader import (
    Dataloader,
)
from tests.fakes.fake_model import make_false_label
from tests.fakes.fake_pytorch_audio_factory import create_empty_tensor


class FakeDataloader(Dataloader):
    def __init__(self, batches: List[PytorchBatch]):
        self._batches = batches

    def get_batches(self) -> Sequence[PytorchBatch]:
        return self._batches


def make_batch():
    features = [create_empty_tensor() for _ in range(3)]
    labels = [make_false_label() for _ in range(3)]
    return PytorchBatch(
        features=features,
        labels=labels,
    )


def make_batches(
    length: int = 3,
):
    return [make_batch() for _ in range(length)]


def make_dataloader(
    batches: List[PytorchBatch] | None = None,
):
    return FakeDataloader(batches=batches or make_batches())
