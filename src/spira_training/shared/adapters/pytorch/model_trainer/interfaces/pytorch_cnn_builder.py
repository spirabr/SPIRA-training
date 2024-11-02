from abc import ABC, abstractmethod
from pydantic import BaseModel
import torch.nn as nn


class PytorchCnnConfig(BaseModel):
    name: str
    fc1_dim: int
    fc2_dim: int


class PytorchCnnBuilder(ABC):
    def __init__(self, config: PytorchCnnConfig, num_features: int):
        self.config = config
        self.num_features = num_features

    @staticmethod
    def define_dropout() -> nn.Dropout:
        return nn.Dropout(p=0.7)

    @abstractmethod
    def build_fc2(self) -> nn.Linear:
        return nn.Linear(self.config.fc1_dim, self.config.fc2_dim)

    @abstractmethod
    def build_fc1(self, conv) -> nn.Linear: ...

    @abstractmethod
    def reshape_x(self, x): ...
