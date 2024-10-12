from abc import ABC, abstractmethod

from spira_training.shared.adapters.model_trainer.pytorch_model_trainer.pytorch_audio import (
    PytorchAudio,
)

from src.spira_training.shared.core.models.audio import Audio


class PytorchAudioFactory(ABC):
    @abstractmethod
    def create_pytorch_from_audio(self, audio: Audio) -> PytorchAudio: ...
