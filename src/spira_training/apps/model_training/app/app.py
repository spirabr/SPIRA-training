from src.spira_training.apps.model_training.app.interfaces.dataset_splitter import (
    DatasetSplitter,
)
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
        dataset_splitter: DatasetSplitter,
        model_trainer: ModelTrainer,
        trained_models_repository: TrainedModelsRepository,
    ):
        self._dataset_repository = dataset_repository
        self._model_trainer = model_trainer
        self._trained_models_repository = trained_models_repository
        self._dataset_splitter = dataset_splitter

    async def execute(
        self,
        dataset_path: str,
        model_storage_path: str,
    ) -> None:
        dataset = await self._dataset_repository.get_dataset(dataset_path)
        splitted_dataset = self._dataset_splitter.split(dataset)
        trained_model = await self._model_trainer.train_model(
            train_dataset=splitted_dataset.train_dataset,
            test_dataset=splitted_dataset.test_dataset,
        )
        await self._trained_models_repository.save_model(
            trained_model, model_storage_path
        )
