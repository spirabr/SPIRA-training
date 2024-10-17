import torch


from typing import NewType

__all__ = ["PytorchWav"]

PytorchWav = NewType("PytorchWav", torch.Tensor)
