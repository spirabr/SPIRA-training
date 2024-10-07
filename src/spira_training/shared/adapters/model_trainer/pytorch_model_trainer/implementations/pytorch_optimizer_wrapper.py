from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.optimizer import (
    Optimizer,
)
import torch


class PytorchOptimizerWrapper(Optimizer):
    def __init__(self, torch_optimizer: torch.optim.Optimizer):
        self.torch_optimizer = torch_optimizer

    def zero_grad(self):
        self.torch_optimizer.zero_grad()

    def step(self):
        return self.torch_optimizer.step()

    def dump_state(self) -> dict:
        return self.torch_optimizer.state_dict()

    def load_state(self, state: dict):
        self.torch_optimizer.load_state_dict(state)
