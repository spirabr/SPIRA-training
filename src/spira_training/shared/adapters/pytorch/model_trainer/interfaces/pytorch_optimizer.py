from abc import ABC
from typing import Generic

from src.spira_training.shared.core.models.model_params import ModelParamsT


class PytorchOptimizer(ABC, Generic[ModelParamsT]):
    def step(self): ...

    def zero_grad(self): ...

    def dump_state(self) -> dict: ...

    def load_state(self, state: dict): ...
