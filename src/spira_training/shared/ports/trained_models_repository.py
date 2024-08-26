from abc import ABC, abstractmethod
from pathlib import Path
from src.spira_training.shared.core.models.trained_model import TrainedModel


class TrainedModelsRepository(ABC):
    @abstractmethod
    async def get_model(self, path: Path) -> TrainedModel:
        pass

    @abstractmethod
    async def save_model(self, model: TrainedModel, path: Path) -> None:
        pass
