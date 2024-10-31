from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_dataloader import (
    PytorchDataloader,
)
from src.spira_training.shared.core.models.dataset import Dataset

from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.dataloader_factory import (
    DataloaderFactory,
)

from tests.fakes.fake_dataloader import FakeDataloader, make_batches


class FakeDataloaderFactory(DataloaderFactory):
    def __init__(self):
        self.dataloader = FakeDataloader(batches=make_batches())

    def with_dataloader(self, dataloader: PytorchDataloader) -> "FakeDataloaderFactory":
        self.dataloader = dataloader
        return self

    def make_dataloader(self, dataset: Dataset) -> PytorchDataloader:
        return self.dataloader
