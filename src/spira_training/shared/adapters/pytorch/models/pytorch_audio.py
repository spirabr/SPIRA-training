from src.spira_training.shared.adapters.pytorch.models.pytorch_wav import (
    PytorchWav,
)
import torch


class PytorchAudio:
    def __init__(self, wav: PytorchWav, sample_rate: int):
        self.wav = wav
        self.sample_rate = sample_rate


def create_empty_wav() -> PytorchWav:
    return PytorchWav(torch.empty(0))
