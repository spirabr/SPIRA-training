from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.scheduler import (
    Scheduler,
)


class FakeScheduler(Scheduler):
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1

    def assert_step_called_times(self, times: int):
        assert (
            self.step_calls == times
        ), f"Expected {times} calls to step, but got {self.step_calls}"
