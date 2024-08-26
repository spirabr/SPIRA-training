from typing import Optional
from src.spira_training.shared.core.interfaces.dataset_splitter import DatasetSplitter
from src.spira_training.shared.core.models.splitted_dataset import SplittedDataset


class FakeDatasetSplitter(DatasetSplitter):
    def __init__(self):
        self._split_result: Optional[SplittedDataset] = None

    def with_split_result(self, split_result: SplittedDataset):
        self._split_result = split_result
        return self

    def split(self, dataset):
        return self._split_result
