from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.implementations.pytorch_dataloader_factory import (
    PytorchDataloaderFactory,
)
from tests.fakes.fake_dataset_repository import make_dataset


def test_train_dataloader_factory_executes():
    sut = PytorchDataloaderFactory(
        batch_size=1,
        dataloader_type="train",
        num_workers=1,
    )

    result = sut.make_dataloader(dataset=make_dataset())

    assert result is not None


def test_test_dataloader_factory_executes():
    sut = PytorchDataloaderFactory(
        batch_size=1,
        dataloader_type="test",
        num_workers=1,
    )

    result = sut.make_dataloader(dataset=make_dataset())

    assert result is not None


def test_get_batches():
    sut = PytorchDataloaderFactory(
        batch_size=1,
        dataloader_type="test",
        num_workers=1,
    )

    result = sut.make_dataloader(dataset=make_dataset())

    assert result.get_batches() is not None
