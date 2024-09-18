from typing import Sequence


from src.spira_training.shared.core.models.dataset import Dataset

from .interfaces.dataloader_factory import DataloaderFactory
from .interfaces.optimizer import Optimizer
from src.spira_training.shared.core.models.batch import Batch
from src.spira_training.shared.ports.model_trainer import ModelTrainer
from src.spira_training.shared.core.models.trained_model import TrainedModel


BaseModel = TrainedModel


class PytorchModelTrainer(ModelTrainer):
    def __init__(
        self,
        base_model: BaseModel,
        optimizer: Optimizer,
        train_dataloader_factory: DataloaderFactory,
        test_dataloader_factory: DataloaderFactory,
    ) -> None:
        self._model = base_model
        self._optimizer = optimizer
        self._train_dataloader_factory = train_dataloader_factory
        self._test_dataloader_factory = test_dataloader_factory

    def train_model(
        self, train_dataset: Dataset, test_dataset: Dataset, epochs: int
    ) -> TrainedModel:
        train_dataloader = self._train_dataloader_factory.make_dataloader(
            dataset=train_dataset
        )
        test_dataloader = self._test_dataloader_factory.make_dataloader(
            dataset=test_dataset
        )

        for _ in range(0, epochs):
            self._execute_training_epoch(
                train_batches=train_dataloader.get_batches(),
                test_batches=test_dataloader.get_batches(),
            )

        return self._model

    def _execute_training_epoch(
        self, train_batches: Sequence[Batch], test_batches: Sequence[Batch]
    ):
        for train_batch in train_batches:
            labels = self._model.predict_batch(train_batch.features)
            self._optimizer.step()
