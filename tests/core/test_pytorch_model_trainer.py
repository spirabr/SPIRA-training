from typing import List
from src.spira_training.shared.core.models.batch import Batch
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
    train_batches: List[Batch]
    test_batches: List[Batch]
    train_dataloader: FakeDataloader
    test_dataloader: FakeDataloader


def make_setup() -> SetupData:
    base_model = FakeModel()
    train_batches = [make_dataset(), make_dataset(), make_dataset()]
    test_batches = [make_dataset(), make_dataset(), make_dataset()]
    train_dataloader = FakeDataloader(batches=train_batches)
    test_dataloader = FakeDataloader(batches=[make_dataset(), make_dataset()])

    return {
        "sut": PytorchModelTrainer(base_model=base_model),
        "base_model": base_model,
        "test_batches": test_batches,
        "train_batches": train_batches,
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
    }


def test_returns_trained_model():
    # Arrange
    validation_feature = make_audio()

    setup = make_setup()
    sut = setup["sut"]
    train_dataloader = setup["train_dataloader"]
    test_dataloader = setup["test_dataloader"]

    # Act
    trained_model = sut.train_model(
        train_dataloader=train_dataloader, test_dataloader=test_dataloader, epochs=1
    )

    # Assert
    prediction_result = trained_model.predict(validation_feature)

    assert Label.has_value(prediction_result.value)


def test_trains_with_each_batch_once():
    # Arrange
    setup = make_setup()
    sut = setup["sut"]
    base_model = setup["base_model"]
    train_batches = setup["train_batches"]
    train_dataloader = setup["train_dataloader"]
    test_dataloader = setup["test_dataloader"]
    # Act
    sut.train_model(
        train_dataloader=train_dataloader, test_dataloader=test_dataloader, epochs=1
    )

    # Assert
    for batch in train_batches:
        for feature in batch.features:
            base_model.assert_predicted_once(feature)


def test_trains_with_each_batch_for_each_epoch():
    # Arrange

    setup = make_setup()
    sut = setup["sut"]
    base_model = setup["base_model"]
    train_batches = setup["train_batches"]
    train_dataloader = setup["train_dataloader"]
    test_dataloader = setup["test_dataloader"]
    epochs = 3

    # Act
    sut.train_model(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=epochs,
    )

    # Assert
    for batch in train_batches:
        for feature in batch.features:
            base_model.assert_predicted_times(feature=feature, times=epochs)
