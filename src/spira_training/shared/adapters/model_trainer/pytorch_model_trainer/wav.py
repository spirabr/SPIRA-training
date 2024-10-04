from typing import NewType

import torch

Wav = NewType("Wav", torch.Tensor)


def create_empty_wav() -> Wav:
    return Wav(torch.empty(0))
