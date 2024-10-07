from abc import abstractmethod

from .pytorch_optimizer_wrapper import PytorchOptimizerWrapper
from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.scheduler import (
    Scheduler,
)
import torch


class PytorchLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    @abstractmethod
    def get_lr(self):
        pass


class NoamLRScheduler(Scheduler, PytorchLRScheduler):
    def __init__(
        self, pytorch_optimizer_wrapper: PytorchOptimizerWrapper, warmup_steps: float
    ):
        super().__init__(pytorch_optimizer_wrapper.torch_optimizer)

        self.warmup_steps = float(warmup_steps)

        self.scheduler = NoamLRScheduler(
            pytorch_optimizer_wrapper, warmup_steps=warmup_steps
        )

    def get_lr(self):
        step = max(self.last_epoch, 1)
        return [
            base_lr
            * self.warmup_steps**0.5
            * min(step * self.warmup_steps**-1.5, step**-0.5)
            for base_lr in self.base_lrs
        ]
