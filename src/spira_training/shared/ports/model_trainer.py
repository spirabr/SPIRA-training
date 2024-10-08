from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.dataset import Dataset

from src.spira_training.shared.core.models.trained_model import TrainedModel


class ModelTrainer(ABC):
    @abstractmethod
    def train_model(
        self, train_dataset: Dataset, test_dataset: Dataset, epochs: int
    ) -> TrainedModel:
        pass
