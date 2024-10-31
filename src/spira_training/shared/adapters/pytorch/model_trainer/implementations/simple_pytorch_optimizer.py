from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_optimizer import (
    PytorchOptimizer,
)
import torch


class SimplePytorchOptimizer(PytorchOptimizer):
    def __init__(self, torch_optimizer: torch.optim.optimizer.Optimizer):
        self.torch_optimizer = torch_optimizer

    def zero_grad(self):
        self.torch_optimizer.zero_grad()

    def step(self):
        return self.torch_optimizer.step()

    def dump_state(self) -> dict:
        return self.torch_optimizer.state_dict()

    def load_state(self, state: dict):
        self.torch_optimizer.load_state_dict(state)
