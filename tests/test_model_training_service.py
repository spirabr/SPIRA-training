from pathlib import Path
import pytest
from src.spira_training.shared.core.models.splitted_dataset import SplittedDataset
from src.spira_training.shared.core.services.model_training_service import (
    ModelTrainingService,
)
from tests.fakes.fake_dataset_repository import FakeDatasetRepository, make_dataset
from tests.fakes.fake_dataset_splitter import FakeDatasetSplitter
from tests.fakes.fake_model_trainer import FakeModelTrainer
from tests.fakes.fake_trained_models_repository import (
    FakeTrainedModelsRepository,
    make_trained_model,
)


def make_sut(
    dataset_repository=None,
    model_trainer=None,
    trained_models_repository=None,
    dataset_splitter=None,
):
    dataset_repository = dataset_repository or FakeDatasetRepository()
    _model_trainer = model_trainer or FakeModelTrainer()
    _trained_models_repository = (
        trained_models_repository or FakeTrainedModelsRepository()
    )
    _dataset_splitter = dataset_splitter or FakeDatasetSplitter()
    return ModelTrainingService(
        dataset_repository=dataset_repository,
        model_trainer=_model_trainer,
        trained_models_repository=_trained_models_repository,
        dataset_splitter=_dataset_splitter,
    )


@pytest.mark.asyncio
async def test_execute():
    # Arrange
    dataset_path = Path("dataset_path")
    trained_model_path = Path("any_model_storage_path")

    base_dataset = make_dataset()
    training_dataset = make_dataset()
    test_dataset = make_dataset()
    trained_model = make_trained_model()

    dataset_repository = FakeDatasetRepository()
    dataset_splitter = FakeDatasetSplitter().with_split_result(
        SplittedDataset(train_dataset=training_dataset, test_dataset=test_dataset)
    )
    trained_models_repository = FakeTrainedModelsRepository()
    model_trainer = FakeModelTrainer().with_train_result(trained_model)

    await dataset_repository.save_dataset(path=dataset_path, dataset=base_dataset)

    sut = make_sut(
        dataset_repository=dataset_repository,
        dataset_splitter=dataset_splitter,
        trained_models_repository=trained_models_repository,
        model_trainer=model_trainer,
    )

    # Act
    await sut.execute(
        trained_model_path=trained_model_path,
        dataset_path=dataset_path,
    )

    # Assert
    assert model_trainer.called_with(
        train_dataset=training_dataset, test_dataset=test_dataset
    )
    saved_model = await trained_models_repository.get_model(path=trained_model_path)
    assert saved_model == trained_model
