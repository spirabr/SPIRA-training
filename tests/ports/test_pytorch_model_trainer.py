from typing_extensions import TypedDict
from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.pytorch_model_trainer import (
    PytorchModelTrainer,
)
from src.spira_training.shared.core.models.dataset import Label
from tests.fakes.fake_audios_repository import make_audio
from tests.fakes.fake_dataloader import FakeDataloader, make_dataloader
from tests.fakes.fake_model import FakeModel
from tests.fakes.fake_optimizer import FakeOptimizer


class SetupData(TypedDict):
    sut: PytorchModelTrainer
    base_model: FakeModel
    train_dataloader: FakeDataloader
    test_dataloader: FakeDataloader
    optimizer: FakeOptimizer


def make_setup() -> SetupData:
    base_model = FakeModel()
    train_dataloader = make_dataloader()
    test_dataloader = make_dataloader()
    optimizer = FakeOptimizer()

    return {
        "sut": PytorchModelTrainer(base_model=base_model, optimizer=optimizer),
        "base_model": base_model,
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "optimizer": optimizer,
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
    train_dataloader = setup["train_dataloader"]
    test_dataloader = setup["test_dataloader"]
    # Act
    sut.train_model(
        train_dataloader=train_dataloader, test_dataloader=test_dataloader, epochs=1
    )

    # Assert
    for batch in train_dataloader.get_batches():
        for feature in batch.features:
            base_model.assert_predicted_once(feature)


def test_trains_with_each_batch_for_each_epoch():
    # Arrange
    setup = make_setup()
    sut = setup["sut"]
    base_model = setup["base_model"]
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
    for batch in train_dataloader.get_batches():
        for feature in batch.features:
            base_model.assert_predicted_times(feature=feature, times=epochs)


def test_executes_optimizer_each_batch():
    # Arrange
    setup = make_setup()
    sut = setup["sut"]
    train_dataloader = setup["train_dataloader"]
    test_dataloader = setup["test_dataloader"]
    optimizer = setup["optimizer"]

    # Act
    sut.train_model(
        train_dataloader=train_dataloader, test_dataloader=test_dataloader, epochs=1
    )

    # Assert
    optimizer.assert_step_called_times(times=len(train_dataloader.get_batches()))
