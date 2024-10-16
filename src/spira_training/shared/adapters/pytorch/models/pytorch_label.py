import torch


from typing import NewType

__all__ = ["PytorchLabel"]

PytorchLabel = NewType("PytorchLabel", torch.Tensor)
