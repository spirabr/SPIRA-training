from abc import ABC, abstractmethod
from typing import List

from src.spira_training.shared.core.models.valid_path import ValidPath


class ValidPathReader(ABC):
    @abstractmethod
    def read_valid_paths(self, path: ValidPath) -> List[ValidPath]:
        pass