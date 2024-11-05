from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_scheduler import (
    PytorchScheduler,
)


def test_optimizer_smoke(scheduler: PytorchScheduler):
    try:
        scheduler.step()

    except Exception as e:
        assert False, f"Failed with exception: {e}"

    assert True
