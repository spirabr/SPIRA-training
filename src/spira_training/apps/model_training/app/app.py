from src.spira_training.apps.model_training.app.interfaces.model_trainer import (
    ModelTrainer,
)
from src.spira_training.shared.ports.dataset_repository import DatasetRepository
from src.spira_training.shared.ports.trained_models_repository import (
    TrainedModelsRepository,
)


class App:
    def __init__(
        self,
        dataset_repository: DatasetRepository,
        model_trainer: ModelTrainer,
        trained_models_repository: TrainedModelsRepository,
    ):
        self._dataset_repository = dataset_repository
        self._model_trainer = model_trainer
        self._trained_models_repository = trained_models_repository

    async def execute(
        self,
        train_dataset_path: str,
        validation_dataset_path: str,
        model_storage_path: str,
    ) -> None:
        train_dataset = await self._dataset_repository.get_dataset(train_dataset_path)
        validation_dataset = await self._dataset_repository.get_dataset(
            validation_dataset_path
        )
        trained_model = await self._model_trainer.train_model(
            train_dataset, validation_dataset
        )
        await self._trained_models_repository.save_model(
            trained_model, model_storage_path
        )
