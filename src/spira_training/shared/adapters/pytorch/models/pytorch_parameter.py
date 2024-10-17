import torch


from typing import NewType


PytorchParameter = NewType("PytorchParameter", torch.nn.Parameter)
