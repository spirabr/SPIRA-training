from spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_scheduler import (
    PytorchScheduler,
)


class FakeScheduler(PytorchScheduler):
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1

    def assert_step_called_times(self, times: int):
        assert (
            self.step_calls == times
        ), f"Expected {times} calls to step, but got {self.step_calls}"
