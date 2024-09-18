from typing import Optional
from src.spira_training.shared.ports.model_trainer import ModelTrainer
from src.spira_training.shared.core.models.dataset import Dataset
from src.spira_training.shared.core.models.trained_model import TrainedModel
from tests.fakes.fake_trained_models_repository import make_trained_model


class FakeModelTrainer(ModelTrainer):
    def __init__(self):
        self.called_with_args = None
        self.train_result: Optional[TrainedModel] = None

    def train_model(
        self, train_dataset: Dataset, test_dataset: Dataset
    ) -> TrainedModel:
        self.called_with_args = {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
        }
        return self.train_result or make_trained_model()

    def called_with(self, train_dataset, test_dataset):
        return self.called_with_args == {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
        }

    def with_train_result(self, trained_model: TrainedModel):
        self.train_result = trained_model
        return self
