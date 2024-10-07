from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.implementations.pytorch_dataloader_factory import (
    PytorchDataloaderFactory,
)
from tests.base_test_model import BaseTestModel
from tests.fakes.fake_dataset_repository import make_dataset
from tests.fakes.fake_wav_factory import FakeWavFactory


class SetupItems(BaseTestModel):
    sut: PytorchDataloaderFactory
    wav_factory: FakeWavFactory


def make_setup(
    dataloader_type: str = "train",
    batch_size: int = 1,
):
    wav_factory = FakeWavFactory()
    sut = PytorchDataloaderFactory(
        batch_size=batch_size,
        dataloader_type=dataloader_type,
        num_workers=1,
        wav_factory=wav_factory,
    )

    return SetupItems(sut=sut, wav_factory=wav_factory)


def test_train_dataloader_factory_executes():
    setup = make_setup(dataloader_type="train")
    sut = setup.sut
    result = sut.make_dataloader(dataset=make_dataset())

    assert result is not None


def test_test_dataloader_factory_executes():
    setup = make_setup(dataloader_type="test")
    sut = setup.sut

    result = sut.make_dataloader(dataset=make_dataset())

    assert result is not None


def test_get_batches():
    setup = make_setup()
    sut = setup.sut

    result = sut.make_dataloader(dataset=make_dataset())

    assert result.get_batches() is not None


def test_uses_wav_factory():
    setup = make_setup()
    sut = setup.sut
    wav_factory = setup.wav_factory
    dataset = make_dataset()

    sut.make_dataloader(dataset=dataset)

    for feature in dataset.features:
        wav_factory.assert_called_with(feature)


def test_dont_shuffle_test_data():
    setup = make_setup(
        batch_size=100,
        dataloader_type="test",
    )
    sut = setup.sut
    dataset = make_dataset(size=100)

    result = sut.make_dataloader(dataset=dataset)

    batches = result.get_batches()
    assert len(batches) == 1

    batch = batches[0]
    for i in range(0, 100):
        ith_batch_label = batch.labels[
            i
        ]  # TODO: Fix this, its currently a tensor but typing says its a Label
        ith_dataset_label = dataset.labels[i].value

        assert (
            ith_batch_label.item() == ith_dataset_label
        ), f"{i}th batch label {ith_batch_label} does not match {i}th dataset label {ith_dataset_label}"
