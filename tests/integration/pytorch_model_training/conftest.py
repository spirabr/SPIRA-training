import pytest
from spira_training.shared.adapters.pytorch.model_trainer.implementations.noam_lr_pytorch_scheduler import (
    NoamLRPytorchScheduler,
)
from spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_cnn_builder.overlapped_features import (
    OverlappedFeaturesPytorchCnnBuilder,
)
from spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_loss_calculator.multiple_loss_calculator import (
    AverageMultipleLossCalculator,
)
from spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_model.inner_torch_model import (
    InnerTorchModel,
)
from spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_checkpoint_manager import (
    PytorchCheckpointManager,
)
from spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_cnn_builder import (
    PytorchCnnBuilder,
    PytorchCnnConfig,
)
from spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_model import (
    PytorchModel,
)
from spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_optimizer import (
    PytorchOptimizer,
)
from spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_scheduler import (
    PytorchScheduler,
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
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_model.simple_pytorch_model import (
    SimplePytorchModel,
)
from tests.unit.fakes.fake_checkpoint_manager import FakeCheckpointManager
from tests.unit.fakes.fake_train_logger import FakeTrainLogger


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
def pytorch_optimizer(inner_torch_optimizer) -> PytorchOptimizer:
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


@pytest.fixture()
def train_logger():
    return FakeTrainLogger()


@pytest.fixture()
def scheduler(
    pytorch_optimizer: PytorchOptimizer,
) -> PytorchScheduler:
    return NoamLRPytorchScheduler(
        pytorch_optimizer_wrapper=pytorch_optimizer,
        warmup_steps=10,
    )


@pytest.fixture()
def checkpoint_manager() -> PytorchCheckpointManager:
    return FakeCheckpointManager()


@pytest.fixture()
def cnn_builder() -> PytorchCnnBuilder:
    return OverlappedFeaturesPytorchCnnBuilder(
        config=PytorchCnnConfig(
            fc1_dim=3,
            fc2_dim=3,
            name="overlapped_features",
        ),
        num_features=11,
    )


@pytest.fixture()
def inner_torch_model(
    cnn_builder: PytorchCnnBuilder,
) -> InnerTorchModel:
    return InnerTorchModel(
        cnn_builder=cnn_builder,
    )


@pytest.fixture()
def base_model(
    inner_torch_model: InnerTorchModel,
) -> PytorchModel:
    return SimplePytorchModel(
        model=inner_torch_model,
    )
