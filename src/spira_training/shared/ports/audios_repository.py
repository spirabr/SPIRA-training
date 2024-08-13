from abc import ABC, abstractmethod
from typing import List
from src.spira_training.shared.models.audio import Audio


class AudiosRepository(ABC):
    @abstractmethod
    def get_audios(self, path: str) -> List[Audio]:
        pass

    @abstractmethod
    def save_audios(self, audios: List[Audio], path: str) -> None:
        pass
