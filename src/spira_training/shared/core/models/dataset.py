from enum import Enum
from typing import List
from src.spira_training.shared.core.models.audio import Audio


class Label(Enum):
    POSITIVE = 1
    NEGATIVE = 0


class Dataset:
    def __init__(self, features: List[Audio], labels: List[Label]):
        self.features = features
        self.labels = labels
