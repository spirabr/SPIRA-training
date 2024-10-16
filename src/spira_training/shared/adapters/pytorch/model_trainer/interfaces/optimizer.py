from abc import ABC
from typing import Generic

from src.spira_training.shared.core.models.model_params import ModelParamsT


class Optimizer(ABC, Generic[ModelParamsT]):
    def step(self): ...
