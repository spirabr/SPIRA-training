from typing import List
from src.spira_training.shared.core.models.transformed_data import TransformedData


class Dataset:
    def __init__(self, data: List[TransformedData]):
        self.data = data
