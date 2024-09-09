import asyncio
from pathlib import Path
from pydantic import BaseModel
from src.spira_training.shared.core.services.model_training_service import (
    ModelTrainingService,
)
from tests.fakes.fake_dataset_repository import FakeDatasetRepository
from tests.fakes.fake_dataset_splitter import FakeDatasetSplitter
from tests.fakes.fake_model_trainer import FakeModelTrainer
from tests.fakes.fake_trained_models_repository import FakeTrainedModelsRepository


class ModelTrainingConfig(BaseModel):
    dataset_path: Path
    trained_model_path: Path


async def main():
    # TODO load config
    config = ModelTrainingConfig(
        dataset_path="dataset_path",
        trained_model_path="trained_model_path",
    )

    # TODO  instantiate the dependencies using configs
    dataset_repository = FakeDatasetRepository()
    dataset_splitter = FakeDatasetSplitter()
    model_trainer = FakeModelTrainer()
    trained_models_repository = FakeTrainedModelsRepository()

    service = ModelTrainingService(
        dataset_repository=dataset_repository,
        dataset_splitter=dataset_splitter,
        model_trainer=model_trainer,
        trained_models_repository=trained_models_repository,
    )

    await service.execute(
        dataset_path=config.dataset_path, trained_model_path=config.trained_model_path
    )


if __name__ == "__main__":
    asyncio.run(main())
