from abc import ABC, abstractmethod

from src.spira_training.shared.core.interfaces.dataloader import Dataloader
from src.spira_training.shared.core.models.trained_model import TrainedModel


class ModelTrainer(ABC):
    @abstractmethod
    def train_model(
        self, train_dataloader: Dataloader, test_dataloader: Dataloader, epochs: int
    ) -> TrainedModel:
        pass
