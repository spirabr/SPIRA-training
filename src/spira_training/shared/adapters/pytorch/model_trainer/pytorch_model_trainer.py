from typing import Sequence

from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_model import (
    PytorchModel,
)

from src.spira_training.shared.adapters.pytorch.models.pytorch_batch import (
    PytorchBatch,
)


from src.spira_training.shared.core.models.event import TestLossEvent, TrainLossEvent

from src.spira_training.shared.ports.train_logger import TrainLogger


from src.spira_training.shared.core.models.dataset import Dataset

from .interfaces.dataloader_factory import DataloaderFactory
from .interfaces.pytorch_optimizer import PytorchOptimizer
from .interfaces.pytorch_loss_calculator import PytorchLossCalculator
from .interfaces.pytorch_scheduler import PytorchScheduler
from .interfaces.pytorch_checkpoint_manager import Checkpoint, PytorchCheckpointManager

from src.spira_training.shared.ports.model_trainer import ModelTrainer
from src.spira_training.shared.core.models.trained_model import TrainedModel


BaseModel = PytorchModel


class PytorchModelTrainer(ModelTrainer):
    def __init__(
        self,
        base_model: BaseModel,
        optimizer: PytorchOptimizer,
        train_dataloader_factory: DataloaderFactory,
        test_dataloader_factory: DataloaderFactory,
        train_loss_calculator: PytorchLossCalculator,
        test_loss_calculator: PytorchLossCalculator,
        train_logger: TrainLogger,
        scheduler: PytorchScheduler,
        checkpoint_manager: PytorchCheckpointManager,
    ) -> None:
        self._model = base_model
        self._optimizer = optimizer
        self._train_dataloader_factory = train_dataloader_factory
        self._test_dataloader_factory = test_dataloader_factory
        self._train_loss_calculator = train_loss_calculator
        self._test_loss_calculator = test_loss_calculator
        self._train_logger = train_logger
        self._scheduler = scheduler
        self._checkpoint_manager = checkpoint_manager

    def train_model(
        self, train_dataset: Dataset, test_dataset: Dataset, epochs: int
    ) -> TrainedModel:
        train_dataloader = self._train_dataloader_factory.make_dataloader(
            dataset=train_dataset
        )
        test_dataloader = self._test_dataloader_factory.make_dataloader(
            dataset=test_dataset
        )

        for epoch in range(0, epochs):
            self._execute_training_epoch(
                train_batches=train_dataloader.get_batches(),
                test_batches=test_dataloader.get_batches(),
                epoch=epoch,
            )

        return self._model

    def _execute_training_epoch(
        self,
        train_batches: Sequence[PytorchBatch],
        test_batches: Sequence[PytorchBatch],
        epoch: int,
    ):
        for train_batch in train_batches:
            self._execute_training_batch(train_batch)

        batches_amount = len(test_batches)

        for batch_index, test_batch in enumerate(test_batches):
            step = (epoch * batches_amount) + batch_index
            self._execute_test_batch(batch=test_batch, step=step, epoch=epoch)

    def _execute_training_batch(self, batch: PytorchBatch):
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
        self._scheduler.step()

    def _execute_test_batch(self, batch: PytorchBatch, step: int, epoch: int):
        predictions = self._model.predict_batch(batch.features)
        loss = self._test_loss_calculator.calculate_loss(
            predictions=predictions, labels=batch.labels
        )
        self._train_logger.log_event(
            TestLossEvent(
                loss=loss,
            )
        )
        self._checkpoint_manager.update_and_save_checkpoint(
            checkpoint=Checkpoint(
                loss=loss,
                step=step,
            )
        )
