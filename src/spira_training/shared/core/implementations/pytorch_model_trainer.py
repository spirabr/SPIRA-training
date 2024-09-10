from src.spira_training.shared.core.interfaces.dataloader import Dataloader
from src.spira_training.shared.core.interfaces.model_trainer import ModelTrainer
from src.spira_training.shared.core.models.trained_model import TrainedModel


class BaseModel(TrainedModel):
    pass


class PytorchModelTrainer(ModelTrainer):
    def __init__(self, base_model: BaseModel) -> None:
        self._model = base_model

    async def train_model(
        self, train_dataloader: Dataloader, test_dataloader: Dataloader, epochs: int
    ) -> TrainedModel:
        for _ in range(0, epochs):
            train_batch = train_dataloader.get_batch()
            labels = self._model.predict_batch(train_batch.features)

        return self._model
