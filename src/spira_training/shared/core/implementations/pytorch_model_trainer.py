from src.spira_training.shared.core.implementations.pytorch_model import PytorchModel
from src.spira_training.shared.core.interfaces.model_trainer import ModelTrainer
from src.spira_training.shared.core.models.dataset import Dataset
from src.spira_training.shared.core.models.trained_model import TrainedModel


class PytorchModelTrainer(ModelTrainer):
    async def train_model(
        self, train_dataset: Dataset, test_dataset: Dataset
    ) -> TrainedModel:
        return PytorchModel()
