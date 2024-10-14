from typing import NewType

import torch

PytorchWav = NewType("PytorchWav", torch.Tensor)
PytorchLabel = NewType("PytorchLabel", torch.Tensor)


class PytorchAudio:
    def __init__(self, wav: PytorchWav, sample_rate: int):
        self.wav = wav
        self.sample_rate = sample_rate


def create_empty_wav() -> PytorchWav:
    return PytorchWav(torch.empty(0))