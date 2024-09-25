from typing import Sequence

from src.spira_training.shared.core.models.event import TestLossEvent, TrainLossEvent

from src.spira_training.shared.ports.train_logger import TrainLogger


from src.spira_training.shared.core.models.dataset import Dataset

from .interfaces.dataloader_factory import DataloaderFactory
from .interfaces.optimizer import Optimizer
from .interfaces.loss_calculator import LossCalculator
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
        train_loss_calculator: LossCalculator,
        test_loss_calculator: LossCalculator,
        train_logger: TrainLogger,
    ) -> None:
        self._model = base_model
        self._optimizer = optimizer
        self._train_dataloader_factory = train_dataloader_factory
        self._test_dataloader_factory = test_dataloader_factory
        self._train_loss_calculator = train_loss_calculator
        self._test_loss_calculator = test_loss_calculator
        self._train_logger = train_logger

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
            self._execute_training_batch(train_batch)

        for test_batch in test_batches:
            self._execute_test_batch(test_batch)

    def _execute_training_batch(self, batch: Batch):
        predictions = self._model.predict_batch(batch.features)
        loss = self._train_loss_calculator.calculate_loss(
            predictions=predictions, labels=batch.labels
        )
        self._train_logger.log_event(
            TrainLossEvent(
                loss=loss,
            )
        )
        self._train_loss_calculator.recalculate_weights()
        self._optimizer.step()

    def _execute_test_batch(self, batch: Batch):
        predictions = self._model.predict_batch(batch.features)
        loss = self._test_loss_calculator.calculate_loss(
            predictions=predictions, labels=batch.labels
        )
        self._train_logger.log_event(
            TestLossEvent(
                loss=loss,
            )
        )
