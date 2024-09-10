from typing import NewType

from src.spira_training.shared.core.models.dataset import Dataset

__all__ = "Batch"


Batch = NewType("Batch", Dataset)
