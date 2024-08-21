import pytest
from src.spira_training.apps.model_training.app.app import App
from src.spira_training.apps.model_training.tests.fake_model_trainer import (
    FakeModelTrainer,
)
from tests.fake_dataset_repository import FakeDatasetRepository, make_dataset
from tests.fake_trained_models_repository import (
    FakeTrainedModelsRepository,
    make_trained_model,
)


def make_sut(
    dataset_repository=None,
    model_trainer=None,
    trained_models_repository=None,
):
    dataset_repository = dataset_repository or FakeDatasetRepository()
    _model_trainer = model_trainer or FakeModelTrainer()
    _trained_models_repository = (
        trained_models_repository or FakeTrainedModelsRepository()
    )
    return App(
        dataset_repository=dataset_repository,
        model_trainer=_model_trainer,
        trained_models_repository=_trained_models_repository,
    )


@pytest.mark.asyncio
async def test_execute():
    # Arrange
    training_dataset = make_dataset()
    validation_dataset = make_dataset()
    dataset_repository = FakeDatasetRepository()
    trained_models_repository = FakeTrainedModelsRepository()

    await dataset_repository.save_dataset(
        path="any_train_path", dataset=training_dataset
    )
    await dataset_repository.save_dataset(
        path="any_validation_path", dataset=validation_dataset
    )
    trained_model = make_trained_model()
    model_trainer = FakeModelTrainer().with_train_result(trained_model)

    sut = make_sut(
        dataset_repository=dataset_repository,
        trained_models_repository=trained_models_repository,
        model_trainer=model_trainer,
    )

    # Act
    await sut.execute(
        train_dataset_path="any_train_path",
        validation_dataset_path="any_validation_path",
        model_storage_path="any_model_storage_path",
    )

    # Assert
    assert model_trainer.called_with(
        train_dataset=training_dataset, validation_dataset=validation_dataset
    )
    saved_model = await trained_models_repository.get_model(
        path="any_model_storage_path"
    )
    assert saved_model == trained_model
