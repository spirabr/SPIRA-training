from src.spira_training.shared.adapters.CSVValidPathReader import CSVValidPathReader
from src.spira_training.shared.adapters.JsonConfigLoader import JsonConfigLoader
from src.spira_training.shared.core.services.feature_engineering_service import FeatureEngineeringService, \
    AudioRepository
from src.spira_training.shared.core.services.randomizer import Randomizer
from tests.fakes.fake_dataset_repository import FakeDatasetRepository


def main():
    config = JsonConfigLoader().load_feature_engineering_config("path/to/config")
    randomizer = Randomizer.initialize_random()
    dataset_repository = FakeDatasetRepository()
    csv_valid_path_reader = CSVValidPathReader()
    audio_repository = AudioRepository()

    service = FeatureEngineeringService(config, randomizer, dataset_repository, csv_valid_path_reader, audio_repository)

    service.execute()

if __name__ == "__main__":
    main()