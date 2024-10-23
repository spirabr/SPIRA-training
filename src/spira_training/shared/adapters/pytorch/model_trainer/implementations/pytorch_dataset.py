from typing import List

from src.spira_training.shared.adapters.pytorch.models.pytorch_label import PytorchLabel

from src.spira_training.shared.adapters.pytorch.models.pytorch_tensor import (
    PytorchTensor,
)
import torch.utils.data


class PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, features: List[PytorchTensor], labels: List[PytorchLabel]):
        self.features = features
        self.labels = labels

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.features):
            raise IndexError("Index out of bounds")
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)
