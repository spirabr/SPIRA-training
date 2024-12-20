from src.spira_training.shared.core.models.path import Path
from src.spira_training.shared.ports.dataset_splitter import DatasetSplitter
from src.spira_training.shared.ports.model_trainer import ModelTrainer
from src.spira_training.shared.ports.dataset_repository import DatasetRepository
from src.spira_training.shared.ports.trained_models_repository import (
    TrainedModelsRepository,
)


class ModelTrainingService:
    def __init__(
        self,
        dataset_repository: DatasetRepository,
        dataset_splitter: DatasetSplitter,
        model_trainer: ModelTrainer,
        trained_models_repository: TrainedModelsRepository,
    ):
        self._dataset_repository = dataset_repository
        self._model_trainer = model_trainer
        self._trained_models_repository = trained_models_repository
        self._dataset_splitter = dataset_splitter

    async def execute(self, dataset_path: Path, trained_model_path: Path) -> None:
        dataset = await self._dataset_repository.get_dataset(dataset_path)
        splitted_dataset = self._dataset_splitter.split(dataset)
        trained_model = self._model_trainer.train_model(
            train_dataset=splitted_dataset.train_dataset,
            test_dataset=splitted_dataset.test_dataset,
            epochs=1,
        )
        await self._trained_models_repository.save_model(
            trained_model, trained_model_path
        )
