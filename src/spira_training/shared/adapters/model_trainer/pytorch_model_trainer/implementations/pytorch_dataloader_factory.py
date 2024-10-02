from .pytorch_dataset import PytorchDataset
from .pytorch_dataloader import PytorchDataloader
from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.dataloader import (
    Dataloader,
)
from src.spira_training.shared.core.models.dataset import Dataset
from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.dataloader_factory import (
    DataloaderFactory,
)


class PytorchDataloaderFactory(DataloaderFactory):
    def __init__(self, batch_size: int, num_workers: int) -> None:
        self._batch_size = batch_size
        self._num_workers = num_workers

    def make_dataloader(self, dataset: Dataset) -> Dataloader:
        pytorch_dataset = PytorchDataset(
            features=dataset.features, labels=dataset.labels
        )
        return PytorchDataloader(
            dataset=pytorch_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )
