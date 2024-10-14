from pathlib import Path
from unittest.mock import patch

import pytest

from src.spira_training.shared.core.services.feature_engineering_service import FeatureEngineeringService
from tests.fakes.fake_dataset_repository import FakeDatasetRepository
from tests.fakes.fake_audios_repository import FakeAudiosRepository
from tests.fakes.fake_pytorch_audio_factory import FakePytorchAudioFactory
from tests.fakes.fake_randomizer import FakeRandomizer
from tests.fakes.fake_feature_engineering_config import make_feature_engineering_config
from tests.fakes.fake_path_validator import FakePathValidator
from tests.fakes.fake_file_reader import FakeFileReader

def make_sut(
    config=None,
    randomizer=None,
    dataset_repository=None,
    audios_repository=None,
    file_reader=None,
    path_validator=None,
    pytorch_audio_factory=None
):
    config = config or make_feature_engineering_config()
    randomizer = randomizer or FakeRandomizer()
    dataset_repository = dataset_repository or FakeDatasetRepository()
    audios_repository = audios_repository or FakeAudiosRepository()
    file_reader = file_reader or FakeFileReader()
    path_validator = path_validator or FakePathValidator()
    pytorch_audio_factory = pytorch_audio_factory or FakePytorchAudioFactory()
    return FeatureEngineeringService(
        config=config,
        randomizer=randomizer,
        dataset_repository=dataset_repository,
        audios_repository=audios_repository,
        file_reader=file_reader,
        path_validator=path_validator,
        pytorch_audio_factory=pytorch_audio_factory
    )

@pytest.mark.asyncio
async def test_execute():
    # Arrange

    save_dataset_path = Path("any_dataset_storage_path")

    config = make_feature_engineering_config()
    randomizer = FakeRandomizer()
    dataset_repository = FakeDatasetRepository()
    audios_repository = FakeAudiosRepository()
    file_reader = FakeFileReader()
    path_validator = FakePathValidator()
    pytorch_audio_factory = FakePytorchAudioFactory()

    sut = make_sut(
        config=config,
        randomizer=randomizer,
        dataset_repository=dataset_repository,
        audios_repository=audios_repository,
        file_reader=file_reader,
        path_validator=path_validator,
        pytorch_audio_factory=pytorch_audio_factory
    )

    # Act
    await sut.execute(save_dataset_path=save_dataset_path)

    # Assert
    assert dataset_repository.save_dataset_called
    assert audios_repository.get_audio_called

@pytest.mark.asyncio
async def test_audio_processor_creation():
    # Arrange
    save_dataset_path = Path("any_dataset_storage_path")
    config = make_feature_engineering_config()
    randomizer = FakeRandomizer()
    dataset_repository = FakeDatasetRepository()
    audios_repository = FakeAudiosRepository()
    file_reader = FakeFileReader()
    path_validator = FakePathValidator()
    pytoch_audio_factory = FakePytorchAudioFactory()

    sut = make_sut(
        config=config,
        randomizer=randomizer,
        dataset_repository=dataset_repository,
        audios_repository=audios_repository,
        file_reader=file_reader,
        path_validator=path_validator,
        pytorch_audio_factory=pytoch_audio_factory
    )

    with patch('src.spira_training.shared.core.services.feature_engineering_service.create_audio_processor') as mock_create_audio_processor:
        # Act
        await sut.execute(save_dataset_path=save_dataset_path)

        # Assert
        mock_create_audio_processor.assert_called_once_with(config.audio_processor, pytoch_audio_factory)
