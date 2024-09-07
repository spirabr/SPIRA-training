from src.spira_training.shared.core.interfaces.model_trainer import ModelTrainer
from src.spira_training.shared.core.models.dataset import Dataset
from src.spira_training.shared.core.models.trained_model import TrainedModel


class BaseModel(TrainedModel):
    pass


class PytorchModelTrainer(ModelTrainer):
    def __init__(self, base_model: BaseModel) -> None:
        self._model = base_model

    async def train_model(
        self, train_dataset: Dataset, test_dataset: Dataset
    ) -> TrainedModel:
        for i in range(len(train_dataset.features)):
            feature = train_dataset.features[i]

            prediction = self._model.predict(feature=feature)

        return self._model
