import torch
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_loss_calculator import (
    PytorchLossCalculator,
)


def test_test_loss_calculator_smoke(test_loss_calculator: PytorchLossCalculator):
    try:
        test_loss_calculator.calculate_loss(
            predictions=torch.tensor([1.0]), labels=torch.tensor([1.0])
        )
        test_loss_calculator.recalculate_weights()
    except Exception as e:
        assert False, f"Failed with exception: {e}"

    assert True
