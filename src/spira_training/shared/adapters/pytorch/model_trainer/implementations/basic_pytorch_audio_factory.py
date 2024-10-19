from spira_training.shared.adapters.pytorch.models.pytorch_audio import PytorchAudio
from spira_training.shared.core.models.audio import Audio
import torch
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_audio_factory import (
    PytorchAudioFactory,
)


class BasicPytorchAudioFactory(PytorchAudioFactory):
    def create_pytorch_from_audio(self, audio: Audio) -> PytorchAudio:
        return PytorchAudio(
            wav=torch.tensor(audio.wav.tensor, dtype=torch.float32),
            sample_rate=audio.sample_rate,
        )
