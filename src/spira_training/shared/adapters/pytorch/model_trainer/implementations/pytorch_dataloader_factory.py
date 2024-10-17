from typing import Literal

from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_audio_factory import (
    PytorchAudioFactory,
)

from .pytorch_dataset import PytorchDataset
from .pytorch_dataloader import PytorchDataloader
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.dataloader import (
    Dataloader,
)
from src.spira_training.shared.core.models.dataset import Dataset
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.dataloader_factory import (
    DataloaderFactory,
)

PytorchDataloaderFactoryType = Literal["train", "test"]


class PytorchDataloaderFactory(DataloaderFactory):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        dataloader_type: PytorchDataloaderFactoryType,
        pytorch_audio_factory: PytorchAudioFactory,
    ) -> None:
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._dataloader_type: PytorchDataloaderFactoryType = dataloader_type
        self._pytorch_audio_factory = pytorch_audio_factory

    def make_dataloader(self, dataset: Dataset) -> Dataloader:
        features = [
            self._pytorch_audio_factory.create_pytorch_from_audio(audio).wav
            for audio in dataset.features
        ]
        labels = [label.value for label in dataset.labels]
        pytorch_dataset = PytorchDataset(
            features=features,
            labels=labels,
        )
        return PytorchDataloader(
            dataset=pytorch_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            dataloader_type=self._dataloader_type,
        )
