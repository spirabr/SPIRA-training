from abc import abstractmethod
from typing import List

from src.spira_training.shared.adapters.pytorch.models.pytorch_parameter import (
    PytorchParameter,
)

from src.spira_training.shared.adapters.pytorch.models.pytorch_tensor import (
    PytorchTensor,
)
from src.spira_training.shared.core.models.trained_model import TrainedModel

from src.spira_training.shared.adapters.pytorch.models.pytorch_label import (
    PytorchLabel,
)


class PytorchModel(TrainedModel):
    @abstractmethod
    def predict(self, feature: PytorchTensor) -> PytorchLabel: ...

    @abstractmethod
    def predict_batch(
        self, features_batch: List[PytorchTensor]
    ) -> List[PytorchLabel]: ...

    @abstractmethod
    def get_parameters(self) -> list[PytorchParameter]: ...
