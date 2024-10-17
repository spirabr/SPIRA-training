from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.dataloader import (
    Dataloader,
)
from src.spira_training.shared.core.models.dataset import Dataset

from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.dataloader_factory import (
    DataloaderFactory,
)

from tests.fakes.fake_dataloader import FakeDataloader, make_batches


class FakeDataloaderFactory(DataloaderFactory):
    def __init__(self):
        self.dataloader = FakeDataloader(batches=make_batches())

    def with_dataloader(self, dataloader: Dataloader) -> "FakeDataloaderFactory":
        self.dataloader = dataloader
        return self

    def make_dataloader(self, dataset: Dataset) -> Dataloader:
        return self.dataloader
