from src.spira_training.shared.core.models.event import EventTypeEnum
from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.pytorch_model_trainer import (
    PytorchModelTrainer,
)
from typing_extensions import TypedDict
from src.spira_training.shared.core.models.dataset import Label
from tests.fakes.fake_audios_repository import make_audio
from tests.fakes.fake_dataloader_factory import FakeDataloaderFactory
from tests.fakes.fake_dataset_repository import make_dataset
from tests.fakes.fake_loss_calculator import FakeLossCalculator, make_loss
from tests.fakes.fake_model import FakeModel
from tests.fakes.fake_optimizer import FakeOptimizer
from tests.fakes.fake_train_logger import FakeTrainLogger


class SetupData(TypedDict):
    sut: PytorchModelTrainer
    base_model: FakeModel
    train_dataloader_factory: FakeDataloaderFactory
    test_dataloader_factory: FakeDataloaderFactory
    optimizer: FakeOptimizer
    train_loss_calculator: FakeLossCalculator
    train_logger: FakeTrainLogger


def make_setup() -> SetupData:
    base_model = FakeModel()
    optimizer = FakeOptimizer()
    train_dataloader_factory = FakeDataloaderFactory()
    test_dataloader_factory = FakeDataloaderFactory()
    train_loss_calculator = FakeLossCalculator()
    train_logger = FakeTrainLogger()

    return {
        "sut": PytorchModelTrainer(
            base_model=base_model,
            optimizer=optimizer,
            train_dataloader_factory=train_dataloader_factory,
            test_dataloader_factory=test_dataloader_factory,
            train_loss_calculator=train_loss_calculator,
            train_logger=train_logger,
        ),
        "base_model": base_model,
        "optimizer": optimizer,
        "train_dataloader_factory": train_dataloader_factory,
        "test_dataloader_factory": test_dataloader_factory,
        "train_loss_calculator": train_loss_calculator,
        "train_logger": train_logger,
    }


def test_returns_trained_model():
    # Arrange
    validation_feature = make_audio()
    setup = make_setup()
    sut = setup["sut"]

    # Act
    trained_model = sut.train_model(
        train_dataset=make_dataset(), test_dataset=make_dataset(), epochs=1
    )

    # Assert
    prediction_result = trained_model.predict(validation_feature)

    assert Label.has_value(prediction_result.value)


def test_trains_with_each_batch_once():
    # Arrange
    setup = make_setup()
    sut = setup["sut"]
    base_model = setup["base_model"]
    train_dataloader_factory = setup["train_dataloader_factory"]
    # Act
    sut.train_model(train_dataset=make_dataset(), test_dataset=make_dataset(), epochs=1)

    # Assert
    for batch in train_dataloader_factory.dataloader.get_batches():
        for feature in batch.features:
            base_model.assert_predicted_once(feature)


def test_trains_with_each_batch_for_each_epoch():
    # Arrange
    setup = make_setup()
    sut = setup["sut"]
    base_model = setup["base_model"]
    train_dataloader_factory = setup["train_dataloader_factory"]
    epochs = 3

    # Act
    sut.train_model(
        train_dataset=make_dataset(),
        test_dataset=make_dataset(),
        epochs=epochs,
    )

    # Assert
    for batch in train_dataloader_factory.dataloader.get_batches():
        for feature in batch.features:
            base_model.assert_predicted_times(feature=feature, times=epochs)


def test_executes_optimizer_each_batch():
    # Arrange
    setup = make_setup()
    sut = setup["sut"]
    train_dataloader_factory = setup["train_dataloader_factory"]
    optimizer = setup["optimizer"]

    # Act
    sut.train_model(train_dataset=make_dataset(), test_dataset=make_dataset(), epochs=1)

    # Assert
    optimizer.assert_step_called_times(
        times=len(train_dataloader_factory.dataloader.get_batches())
    )


def test_logs_loss_each_batch():
    # Arrange
    setup = make_setup()
    sut = setup["sut"]
    train_dataloader_factory = setup["train_dataloader_factory"]
    train_logger = setup["train_logger"]
    train_loss_calculator = setup["train_loss_calculator"]
    fixed_loss = make_loss()
    train_loss_calculator.with_fixed_loss(fixed_loss)

    # Act
    sut.train_model(train_dataset=make_dataset(), test_dataset=make_dataset(), epochs=1)

    # Assert
    batch_count = len(train_dataloader_factory.dataloader.get_batches())
    train_loss_events = list(
        filter(
            lambda event: EventTypeEnum.TRAIN_LOSS == event.type,
            train_logger.logged_events,
        )
    )
    assert len(train_loss_events) == batch_count
    for event in train_loss_events:
        assert event.loss == fixed_loss
