from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_optimizer import (
    PytorchOptimizer,
)


def test_optimizer_smoke(pytorch_optimizer: PytorchOptimizer):
    try:
        pytorch_optimizer.step()
        pytorch_optimizer.zero_grad()
        state = pytorch_optimizer.dump_state()
        pytorch_optimizer.load_state(state)
    except Exception as e:
        assert False, f"Failed with exception: {e}"

    assert True
