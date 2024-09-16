from src.spira_training.shared.core.interfaces.optimizer import Optimizer


class FakeOptimizer(Optimizer):
    def __init__(self) -> None:
        self._times_step_called = 0

    def step(self):
        self._times_step_called += 1

    def assert_step_called_times(self, times: int):
        assert (
            self._times_step_called == times
        ), f"Expected step to be called {times} times, but was {self._times_step_called}"
