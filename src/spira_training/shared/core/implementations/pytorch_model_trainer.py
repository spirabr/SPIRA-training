from typing import Sequence

from src.spira_training.shared.core.interfaces.optimizer import Optimizer
from src.spira_training.shared.core.models.batch import Batch
from src.spira_training.shared.core.interfaces.dataloader import Dataloader
from src.spira_training.shared.core.interfaces.model_trainer import ModelTrainer
from src.spira_training.shared.core.models.trained_model import TrainedModel


BaseModel = TrainedModel


class PytorchModelTrainer(ModelTrainer):
    def __init__(self, base_model: BaseModel, optimizer: Optimizer) -> None:
        self._model = base_model
        self._optimizer = optimizer

    def train_model(
        self, train_dataloader: Dataloader, test_dataloader: Dataloader, epochs: int
    ) -> TrainedModel:
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
