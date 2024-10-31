from abc import ABC, abstractmethod

from src.spira_training.shared.adapters.pytorch.models.pytorch_tensor import (
    PytorchTensor,
)


from src.spira_training.shared.core.models.audio import Audio


class PytorchTensorFactory(ABC):
    @abstractmethod
    def create_tensor_from_audio(self, audio: Audio) -> PytorchTensor: ...
