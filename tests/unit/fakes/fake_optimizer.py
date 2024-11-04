from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_optimizer import (
    PytorchOptimizer,
)


class FakeOptimizer(PytorchOptimizer):
    def __init__(self) -> None:
        self._times_step_called = 0

    def step(self):
        self._times_step_called += 1

    def assert_step_called_times(self, times: int):
        assert (
            self._times_step_called == times
        ), f"Expected step to be called {times} times, but was {self._times_step_called}"

    def load_state(self, state: dict): ...

    def dump_state(self) -> dict: ...

    def zero_grad(self): ...
