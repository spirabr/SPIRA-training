import torch


from typing import NewType

__all__ = ["PytorchTensor"]

PytorchTensor = NewType("PytorchTensor", torch.Tensor)
