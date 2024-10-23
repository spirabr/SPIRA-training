from spira_training.shared.adapters.pytorch.models.pytorch_tensor import PytorchTensor
import torch
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_audio_factory import (
    PytorchTensorFactory,
)


class FakePytorchTensorFactory(PytorchTensorFactory):
    def __init__(self) -> None:
        self.call_args = []

    def create_tensor_from_audio(self, audio: Audio) -> PytorchTensor:
        self.call_args.append(audio)
        return create_empty_tensor()

    def assert_called_with(self, audio: Audio):
        assert audio in self.call_args, f"Expected {audio} to be in {self.call_args}"


def create_empty_tensor() -> PytorchTensor:
    return PytorchTensor(torch.empty(0))
