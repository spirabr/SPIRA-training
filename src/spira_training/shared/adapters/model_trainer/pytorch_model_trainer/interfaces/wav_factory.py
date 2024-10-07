from abc import ABC, abstractmethod

from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.wav import (
    Wav,
)

from src.spira_training.shared.core.models.audio import Audio


class WavFactory(ABC):
    @abstractmethod
    def create_wav_from_audio(self, audio: Audio) -> Wav: ...
