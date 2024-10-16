from abc import abstractmethod
from typing import List

from src.spira_training.shared.adapters.pytorch.models.pytorch_parameter import (
    PytorchParameter,
)

from src.spira_training.shared.adapters.pytorch.models.pytorch_wav import (
    PytorchWav,
)
from src.spira_training.shared.core.models.trained_model import TrainedModel

from src.spira_training.shared.adapters.pytorch.models.pytorch_label import (
    PytorchLabel,
)


class PytorchModel(TrainedModel):
    @abstractmethod
    def predict(self, feature: PytorchWav) -> PytorchLabel: ...

    @abstractmethod
    def predict_batch(self, features_batch: List[PytorchWav]) -> List[PytorchLabel]: ...

    @abstractmethod
    def get_parameters(self) -> list[PytorchParameter]: ...
