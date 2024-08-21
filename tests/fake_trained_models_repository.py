from src.spira_training.shared.core.models.trained_model import TrainedModel
from src.spira_training.shared.ports.trained_models_repository import (
    TrainedModelsRepository,
)


class FakeTrainedModelsRepository(TrainedModelsRepository):
    def __init__(self):
        self.models = {}

    async def get_model(self, path: str) -> TrainedModel:
        return self.models.get(path, None)

    async def save_model(self, model: TrainedModel, path: str) -> None:
        self.models[path] = model
        return None


def make_trained_model():
    return TrainedModel()