from typing_extensions import TypedDict
import pytest
from src.spira_training.shared.core.implementations.pytorch_model_trainer import (
    PytorchModelTrainer,
)
from src.spira_training.shared.core.models.dataset import Label
from tests.fakes.fake_audios_repository import make_audio
from tests.fakes.fake_dataset_repository import make_dataset
from tests.fakes.fake_model import FakeModel


class SetupData(TypedDict):
    sut: PytorchModelTrainer
    base_model: FakeModel


def make_setup() -> SetupData:
    base_model = FakeModel()
    return {"sut": PytorchModelTrainer(base_model=base_model), "base_model": base_model}


@pytest.mark.asyncio
async def test_returns_trained_model():
    # Arrange
    train_dataset = make_dataset()
    test_dataset = make_dataset()
    validation_feature = make_audio()

    setup = make_setup()
    sut = setup["sut"]
    # Act
    trained_model = await sut.train_model(
        train_dataset=train_dataset, test_dataset=test_dataset
    )

    # Assert
    prediction_result = trained_model.predict(validation_feature)

    assert Label.has_value(prediction_result.value)


@pytest.mark.asyncio
async def test_uses_all_train_dataset_once():
    # Arrange
    train_dataset = make_dataset()
    test_dataset = make_dataset()

    setup = make_setup()
    sut = setup["sut"]
    base_model = setup["base_model"]

    # Act
    await sut.train_model(train_dataset=train_dataset, test_dataset=test_dataset)

    # Assert
    assert len(train_dataset.features) > 0
    for feature in train_dataset.features:
        base_model.assert_predicted_once(feature)
