from pathlib import Path

import pytest
from src.spira_training.shared.core.services.feature_engineering_service import FeatureEngineeringService
from tests.fakes.fake_dataset_repository import FakeDatasetRepository
from tests.fakes.fake_audios_repository import FakeAudiosRepository
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
):
    config = config or make_feature_engineering_config()
    randomizer = randomizer or FakeRandomizer()
    dataset_repository = dataset_repository or FakeDatasetRepository()
    audios_repository = audios_repository or FakeAudiosRepository()
    file_reader = file_reader or FakeFileReader()
    path_validator = path_validator or FakePathValidator()
    return FeatureEngineeringService(
        config=config,
        randomizer=randomizer,
        dataset_repository=dataset_repository,
        audios_repository=audios_repository,
        file_reader=file_reader,
        path_validator=path_validator,
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

    sut = make_sut(
        config=config,
        randomizer=randomizer,
        dataset_repository=dataset_repository,
        audios_repository=audios_repository,
        file_reader=file_reader,
        path_validator=path_validator,
    )

    # Act
    await sut.execute(save_dataset_path=save_dataset_path)

    # Assert
    assert dataset_repository.save_dataset_called
    assert audios_repository.get_audio_called