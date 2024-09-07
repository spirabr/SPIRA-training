from typing import List
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.enum import BaseEnum


class Label(BaseEnum):
    POSITIVE = 1
    NEGATIVE = 0


class Dataset:
    def __init__(self, features: List[Audio], labels: List[Label]):
        self.features = features
        self.labels = labels
