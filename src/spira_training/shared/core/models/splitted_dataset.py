from src.spira_training.shared.core.models.dataset import Dataset


class SplittedDataset:
    def __init__(self, train_dataset: Dataset, test_dataset: Dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
