from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.config import Config


class ConfigLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> Config:
        pass