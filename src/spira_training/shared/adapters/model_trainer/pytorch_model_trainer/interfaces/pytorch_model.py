from abc import abstractmethod
from typing import List, NewType

import torch
from src.spira_training.shared.core.models.trained_model import TrainedModel

from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.pytorch_audio import (
    PytorchWav,
    PytorchLabel,
)

PytorchParameter = NewType("PytorchParameter", torch.nn.Parameter)


class PytorchModel(TrainedModel):
    @abstractmethod
    def predict(self, feature: PytorchWav) -> PytorchLabel: ...

    @abstractmethod
    def predict_batch(self, features_batch: List[PytorchWav]) -> List[PytorchLabel]: ...

    @abstractmethod
    def get_parameters(self) -> list[PytorchParameter]: ...
