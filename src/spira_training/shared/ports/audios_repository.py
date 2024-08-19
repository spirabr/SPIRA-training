from abc import ABC, abstractmethod
from typing import List
from src.spira_training.shared.core.models.audio import Audio


class AudiosRepository(ABC):
    @abstractmethod
    def get_audios(self, path: str) -> List[Audio]:
        pass
