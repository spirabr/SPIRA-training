import asyncio
from pathlib import Path

from src.spira_training.shared.adapters.json_config_loader import JsonConfigLoader
from src.spira_training.shared.core.services.feature_engineering_service import FeatureEngineeringService
from src.spira_training.shared.core.services.randomizer import Randomizer
from tests.fakes.fake_dataset_repository import FakeDatasetRepository
from tests.fakes.fake_audios_repository import FakeAudiosRepository
from tests.fakes.fake_path_validator import FakePathValidator
from tests.fakes.fake_file_reader import FakeFileReader

async def main():
    config = JsonConfigLoader().load_feature_engineering_config("path/to/config")

    # TODO - Instantiate the real dependencies
    randomizer = Randomizer.initialize_random()
    dataset_repository = FakeDatasetRepository()
    audios_repository = FakeAudiosRepository()
    file_reader = FakeFileReader()
    path_validator = FakePathValidator()

    service = FeatureEngineeringService(
        config=config,
        randomizer=randomizer,
        dataset_repository=dataset_repository,
        audios_repository=audios_repository,
        file_reader=file_reader,
        path_validator=path_validator,
    )

    # TODO - Get the bucket name to save the dataset
    save_dataset_path = Path("any_model_storage_path")

    await service.execute(save_dataset_path=save_dataset_path)

if __name__ == "__main__":
    asyncio.run(main())