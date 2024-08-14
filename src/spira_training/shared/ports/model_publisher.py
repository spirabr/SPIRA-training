from abc import ABC, abstractmethod
from src.spira_training.shared.core.models.trained_model import TrainedModel


class ModelPublisher(ABC):
    @abstractmethod
    def publish_model(self, model: TrainedModel) -> None:
        pass
