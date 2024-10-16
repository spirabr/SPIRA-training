from abc import ABC, abstractmethod

from src.spira_training.shared.adapters.pytorch.models.pytorch_audio import PytorchAudio

from src.spira_training.shared.core.models.audio import Audio


class PytorchAudioFactory(ABC):
    @abstractmethod
    def create_pytorch_from_audio(self, audio: Audio) -> PytorchAudio: ...
