from typing import List, Sequence
from src.spira_training.shared.core.models.batch import Batch
from src.spira_training.shared.core.interfaces.dataloader import Dataloader


class FakeDataloader(Dataloader):
    def __init__(self, batches: List[Batch]):
        self._batches = batches

    def get_batches(self) -> Sequence[Batch]:
        return self._batches
