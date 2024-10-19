from spira_training.shared.adapters.pytorch.models.pytorch_tensor import PytorchTensor
import torch
from src.spira_training.shared.adapters.pytorch.models.pytorch_audio import (
    PytorchAudio,
)
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_audio_factory import (
    PytorchAudioFactory,
)


class FakePytorchAudioFactory(PytorchAudioFactory):
    def __init__(self) -> None:
        self.call_args = []

    def create_pytorch_from_audio(self, audio: Audio) -> PytorchAudio:
        self.call_args.append(audio)
        return PytorchAudio(wav=create_empty_tensor(), sample_rate=0)

    def assert_called_with(self, audio: Audio):
        assert audio in self.call_args, f"Expected {audio} to be in {self.call_args}"


def create_empty_tensor() -> PytorchTensor:
    return PytorchTensor(torch.empty(0))
