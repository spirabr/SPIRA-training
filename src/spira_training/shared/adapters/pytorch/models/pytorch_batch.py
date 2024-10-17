from typing import List

from src.spira_training.shared.adapters.pytorch.models.pytorch_wav import (
    PytorchWav,
)

from src.spira_training.shared.adapters.pytorch.models.pytorch_label import (
    PytorchLabel,
)


class PytorchBatch:
    def __init__(self, features: List[PytorchWav], labels: List[PytorchLabel]) -> None:
        self.features = features
        self.labels = labels
