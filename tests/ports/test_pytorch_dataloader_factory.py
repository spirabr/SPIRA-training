from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.implementations.pytorch_dataloader import (
    PytorchDataloaderType,
)
from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.implementations.pytorch_dataloader_factory import (
    PytorchDataloaderFactory,
)
from tests.base_test_model import BaseTestModel
from tests.fakes.fake_dataset_repository import make_dataset
from tests.fakes.fake_pytorch_audio_factory import FakePytorchAudioFactory


class SetupItems(BaseTestModel):
    sut: PytorchDataloaderFactory
    pytorch_audio_factory: FakePytorchAudioFactory


def make_setup(
    dataloader_type: PytorchDataloaderType = "train",
    batch_size: int = 1,
):
    pytorch_audio_factory = FakePytorchAudioFactory()
    sut = PytorchDataloaderFactory(
        batch_size=batch_size,
        dataloader_type=dataloader_type,
        num_workers=1,
        pytorch_audio_factory=pytorch_audio_factory,
    )

    return SetupItems(sut=sut, pytorch_audio_factory=pytorch_audio_factory)


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


def test_uses_pytorch_audio_factory():
    setup = make_setup()
    sut = setup.sut
    pytorch_audio_factory = setup.pytorch_audio_factory
    dataset = make_dataset()

    sut.make_dataloader(dataset=dataset)

    for feature in dataset.features:
        pytorch_audio_factory.assert_called_with(feature)


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
        ith_batch_label = batch.labels[i]
        ith_dataset_label = dataset.labels[i].value

        assert (
            ith_batch_label == ith_dataset_label
        ), f"{i}th batch label {ith_batch_label} does not match {i}th dataset label {ith_dataset_label}"
