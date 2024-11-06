from abc import ABC, abstractmethod
from typing import cast

from spira_training.shared.adapters.pytorch.models.pytorch_label import PytorchLabel
from spira_training.shared.adapters.pytorch.models.pytorch_tensor import PytorchTensor
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from typing_extensions import Self


__all__ = ["SingleLossCalculator", "BCELossCalculator", "ClipBCELossCalculator"]


class SingleLossCalculator(ABC):
    @abstractmethod
    def calculate_single_loss_tensor(
        self, prediction: PytorchLabel, label: PytorchLabel
    ) -> PytorchTensor: ...

    @abstractmethod
    def recalculate_weights(self): ...

    @abstractmethod
    def clone(self) -> Self: ...


class BCELossCalculator(SingleLossCalculator):
    def __init__(self, reduction: str):
        self.reduction = reduction
        self.bce_loss = nn.BCELoss(reduction=reduction)

    def calculate_single_loss_tensor(
        self, prediction: PytorchLabel, label: PytorchLabel
    ) -> PytorchTensor:
        tensor_loss = self.bce_loss(prediction, label)
        return tensor_loss

    def recalculate_weights(self):
        self.bce_loss.reduction = self.reduction

    def clone(self) -> Self:
        return cast(Self, BCELossCalculator(reduction=self.reduction))


class ClipBCELossCalculator(SingleLossCalculator):
    def calculate_single_loss_tensor(
        self, prediction: PytorchLabel, label: PytorchLabel
    ) -> PytorchTensor:
        return PytorchTensor(binary_cross_entropy(prediction, label))

    def recalculate_weights(self): ...

    def clone(self) -> Self:
        return cast(Self, ClipBCELossCalculator())
