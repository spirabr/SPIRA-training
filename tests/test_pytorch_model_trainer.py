import pytest
from src.spira_training.shared.core.implementations.pytorch_model_trainer import (
    PytorchModelTrainer,
)
from src.spira_training.shared.core.models.dataset import Label
from tests.fakes.fake_audios_repository import make_audio
from tests.fakes.fake_dataset_repository import make_dataset


def make_sut():
    return PytorchModelTrainer()


@pytest.mark.asyncio
async def test_returns_trained_model():
    # Arrange
    train_dataset = make_dataset()
    test_dataset = make_dataset()
    validation_feature = make_audio()

    sut = make_sut()

    # Act
    trained_model = await sut.train_model(
        train_dataset=train_dataset, test_dataset=test_dataset
    )

    # Assert
    prediction_result = trained_model.predict(validation_feature)

    assert Label.has_value(prediction_result.value)
