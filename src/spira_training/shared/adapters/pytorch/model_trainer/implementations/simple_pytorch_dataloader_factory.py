from typing import Literal

from src.spira_training.shared.adapters.pytorch.models.pytorch_dataset import (
    PytorchDataset,
)
import torch

from src.spira_training.shared.adapters.pytorch.models.pytorch_label import PytorchLabel

from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_tensor_factory import (
    PytorchTensorFactory,
)

from .simple_pytorch_dataloader import SimplePytorchDataloader
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_dataloader import (
    PytorchDataloader,
)
from src.spira_training.shared.core.models.dataset import Dataset
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_dataloader_factory import (
    PytorchDataloaderFactory,
)

PytorchDataloaderFactoryType = Literal["train", "test"]


class SimplePytorchDataloaderFactory(PytorchDataloaderFactory):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        dataloader_type: PytorchDataloaderFactoryType,
        pytorch_tensor_factory: PytorchTensorFactory,
    ) -> None:
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._dataloader_type: PytorchDataloaderFactoryType = dataloader_type
        self._pytorch_tensor_factory = pytorch_tensor_factory

    def make_dataloader(self, dataset: Dataset) -> PytorchDataloader:
        features = [
            self._pytorch_tensor_factory.create_tensor_from_audio(audio)
            for audio in dataset.features
        ]
        labels = [PytorchLabel(torch.tensor(label.value)) for label in dataset.labels]
        pytorch_dataset = PytorchDataset(
            features=features,
            labels=labels,
        )
        return SimplePytorchDataloader(
            dataset=pytorch_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            dataloader_type=self._dataloader_type,
        )
