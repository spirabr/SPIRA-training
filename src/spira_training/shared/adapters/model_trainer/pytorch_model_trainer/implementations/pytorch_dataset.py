from typing import List
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.dataset import Dataset, Label
import torch.utils.data


class PytorchDataset(torch.utils.data.Dataset, Dataset):
    def __init__(self, features: List[Audio], labels: list[Label]):
        self.features = features
        self.labels = labels

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.features):
            raise IndexError("Index out of bounds")
        return self.features[index], self.labels[index]
