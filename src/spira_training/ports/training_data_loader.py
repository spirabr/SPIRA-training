
from abc import ABC, abstractmethod
from domain.model.training_data import TrainingData


class TrainingDataLoader(ABC):
    @abstractmethod
    def load_data(self) -> TrainingData:
        pass
    