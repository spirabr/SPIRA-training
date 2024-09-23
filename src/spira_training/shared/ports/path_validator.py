from abc import ABC, abstractmethod
from pathlib import Path

from src.spira_training.shared.core.models.valid_path import ValidPath


class PathValidator(ABC):
    @abstractmethod
    def validate_path(self, path: Path) -> ValidPath:
        pass