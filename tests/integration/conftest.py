import pytest
import torch
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.simple_pytorch_optimizer import (
    SimplePytorchOptimizer,
)
from torch.optim import Adam


@pytest.fixture()
def inner_torch_optimizer():
    inner_torch_optimizer = Adam(
        params=[
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([2.0], requires_grad=True),
        ],
        lr=0.3,
        weight_decay=0.1,
    )
    return inner_torch_optimizer


@pytest.fixture()
def pytorch_optimizer(inner_torch_optimizer):
    return SimplePytorchOptimizer(torch_optimizer=inner_torch_optimizer)
