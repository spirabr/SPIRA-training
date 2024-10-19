from src.spira_training.shared.adapters.pytorch.models.pytorch_tensor import (
    PytorchTensor,
)


class PytorchAudio:
    def __init__(self, wav: PytorchTensor, sample_rate: int):
        self.wav = wav
        self.sample_rate = sample_rate
