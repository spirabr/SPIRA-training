from typing import List
from src.spira_training.shared.core.models.batch import Batch
from src.spira_training.shared.core.interfaces.dataloader import Dataloader


class FakeDataloader(Dataloader):
    def __init__(self, batches: List[Batch]):
        self._batches_read = 0
        self._batches = batches

    def get_batch(self) -> Batch:
        old_batches_size = self._batches_read
        self._batches_read += 1
        return self._batches[old_batches_size]
