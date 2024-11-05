import pytest
import torch
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_model import (
    PytorchModel,
)


# this doesnt work
@pytest.mark.xfail
def test_optimizer_smoke(base_model: PytorchModel):
    try:
        state = base_model.dump_state()
        base_model.load_state(state)
        base_model.get_parameters()
        input_tensor = torch.tensor(
            [[1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0]]
        ).view(13, 1)  # Reshape to 13x1
        base_model.predict(
            feature=input_tensor,
        )

    except Exception as e:
        assert False, f"Failed with exception: {e}"

    assert True
