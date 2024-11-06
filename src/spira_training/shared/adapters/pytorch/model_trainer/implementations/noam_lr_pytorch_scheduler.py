from typing import List

from spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_scheduler import (
    PytorchScheduler,
)

from .simple_pytorch_optimizer import SimplePytorchOptimizer
import torch


class LrPytorchScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer):
        super().__init__(optimizer)

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class NoamLRPytorchScheduler(PytorchScheduler):
    def __init__(
        self, pytorch_optimizer_wrapper: SimplePytorchOptimizer, warmup_steps: float
    ):
        self.warmup_steps = float(warmup_steps)

        self.scheduler = LrPytorchScheduler(
            optimizer=pytorch_optimizer_wrapper.torch_optimizer
        )

    def get_lr(self) -> List[float]:
        step = max(self.scheduler.last_epoch, 1)
        return [
            base_lr
            * self.warmup_steps**0.5
            * min(step * self.warmup_steps**-1.5, step**-0.5)
            for base_lr in self.scheduler.base_lrs
        ]

    def step(self, epoch: int | None = None):
        self.scheduler.step(epoch)
