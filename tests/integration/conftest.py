import pytest
from spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_loss_calculator.multiple_loss_calculator import (
    AverageMultipleLossCalculator,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.simple_pytorch_dataloader_factory import (
    SimplePytorchDataloaderFactory,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_loss_calculator import (
    PytorchLossCalculator,
)
import torch
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.simple_pytorch_optimizer import (
    SimplePytorchOptimizer,
)
from torch.optim import Adam
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_loss_calculator.single_loss_calculator import (
    BCELossCalculator,
    SingleLossCalculator,
)


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


@pytest.fixture()
def train_dataloader_factory() -> SimplePytorchDataloaderFactory:
    return SimplePytorchDataloaderFactory(
        batch_size=1,
        dataloader_type="train",
        num_workers=1,
        pytorch_tensor_factory=None,
    )


@pytest.fixture()
def test_dataloader_factory() -> SimplePytorchDataloaderFactory:
    return SimplePytorchDataloaderFactory(
        batch_size=1,
        dataloader_type="test",
        num_workers=1,
        pytorch_tensor_factory=None,
    )


@pytest.fixture()
def single_train_loss_calculator() -> SingleLossCalculator:
    return BCELossCalculator(
        reduction="none",
    )


@pytest.fixture()
def train_loss_calculator(
    single_train_loss_calculator: SingleLossCalculator,
) -> PytorchLossCalculator:
    return AverageMultipleLossCalculator(
        single_loss_calculator=single_train_loss_calculator,
    )


@pytest.fixture()
def single_test_loss_calculator() -> SingleLossCalculator:
    return BCELossCalculator(
        reduction="sum",
    )


@pytest.fixture()
def test_loss_calculator(
    single_test_loss_calculator: SingleLossCalculator,
) -> PytorchLossCalculator:
    return AverageMultipleLossCalculator(
        single_loss_calculator=single_test_loss_calculator,
    )
