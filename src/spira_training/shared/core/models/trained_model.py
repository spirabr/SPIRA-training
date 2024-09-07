from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.dataset import Label


class TrainedModel(ABC):
    @abstractmethod
    def predict(self, feature: Audio) -> Label:
        pass
