from typing import List

from src.spira_training.shared.adapters.pytorch.models.pytorch_tensor import (
    PytorchTensor,
)

from src.spira_training.shared.adapters.pytorch.models.pytorch_label import (
    PytorchLabel,
)


class PytorchBatch:
    def __init__(
        self, features: List[PytorchTensor], labels: List[PytorchLabel]
    ) -> None:
        self.features = features
        self.labels = labels
