from typing import Literal, Sequence

from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.models.pytorch_batch import (
    PytorchBatch,
)

from .pytorch_dataset import PytorchDataset
from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.dataloader import (
    Dataloader,
)
from torch.utils.data import DataLoader as PytorchDataLoader


PytorchDataloaderType = Literal["train", "test"]


class PytorchDataloader(Dataloader):
    def __init__(
        self,
        dataset: PytorchDataset,
        batch_size: int,
        num_workers: int,
        dataloader_type: PytorchDataloaderType,
    ) -> None:
        if dataloader_type == "train":
            self._py_dataloader = PytorchDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                sampler=None,
            )
        else:
            self._py_dataloader = PytorchDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )

        self._dataloader_type = dataloader_type

    def get_batches(self) -> Sequence[PytorchBatch]:
        output: Sequence[PytorchBatch] = []

        for py_batch in self._py_dataloader:
            output.append(PytorchBatch(features=py_batch[0], labels=py_batch[1]))

        return output
