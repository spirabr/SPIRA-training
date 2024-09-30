from typing import NewType

import torch

Wav = NewType("Wav", torch.Tensor)
