from typing_extensions import TypedDict
from src.spira_training.shared.core.implementations.pytorch_model_trainer import (
    PytorchModelTrainer,
)
from src.spira_training.shared.core.models.dataset import Label
from tests.fakes.fake_audios_repository import make_audio
from tests.fakes.fake_dataloader import FakeDataloader
from tests.fakes.fake_dataset_repository import make_dataset
from tests.fakes.fake_model import FakeModel


class SetupData(TypedDict):
    sut: PytorchModelTrainer
    base_model: FakeModel


def make_setup() -> SetupData:
    base_model = FakeModel()
    return {"sut": PytorchModelTrainer(base_model=base_model), "base_model": base_model}


def test_returns_trained_model():
    # Arrange
    train_dataloader = FakeDataloader(batches=[make_dataset()])
    test_dataloader = FakeDataloader(batches=[make_dataset()])
    validation_feature = make_audio()

    setup = make_setup()
    sut = setup["sut"]
    # Act
    trained_model = sut.train_model(
        train_dataloader=train_dataloader, test_dataloader=test_dataloader, epochs=1
    )

    # Assert
    prediction_result = trained_model.predict(validation_feature)

    assert Label.has_value(prediction_result.value)


def test_trains_with_first_batch():
    # Arrange
    train_batches = [make_dataset()]
    train_dataloader = FakeDataloader(batches=train_batches)
    test_dataloader = FakeDataloader(batches=[make_dataset()])
    setup = make_setup()
    sut = setup["sut"]
    base_model = setup["base_model"]

    # Act
    sut.train_model(
        train_dataloader=train_dataloader, test_dataloader=test_dataloader, epochs=1
    )

    # Assert
    for feature in train_batches[0].features:
        base_model.assert_predicted_once(feature)


def test_trains_with_batch_for_each_epoch():
    # Arrange
    train_batches = [make_dataset(), make_dataset(), make_dataset()]
    train_dataloader = FakeDataloader(batches=train_batches)
    test_dataloader = FakeDataloader(
        batches=[make_dataset(), make_dataset(), make_dataset()]
    )
    epochs = 3

    setup = make_setup()
    sut = setup["sut"]
    base_model = setup["base_model"]

    # Act
    sut.train_model(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=epochs,
    )

    # Assert
    for batch in train_batches:
        for feature in batch.features:
            base_model.assert_predicted_once(feature)
