from sklearn.model_selection import train_test_split
from src.spira_training.shared.core.models.dataset import Dataset
from src.spira_training.shared.core.models.splitted_dataset import SplittedDataset
from src.spira_training.shared.core.interfaces.dataset_splitter import DatasetSplitter


class SkDatasetSplitter(DatasetSplitter):
    def split(self, dataset: Dataset) -> SplittedDataset:
        X_train, X_test, y_train, y_test = train_test_split(
            X=dataset.features,
            y=dataset.labels,
            train_size=0.6,
            test_size=0.4,
            random_state=42,
        )
        return SplittedDataset(
            train_dataset=Dataset(features=X_train, labels=y_train),
            test_dataset=Dataset(features=X_test, labels=y_test),
        )
