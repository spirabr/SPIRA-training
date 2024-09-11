from src.spira_training.shared.adapters.JsonConfigLoader import JsonConfigLoader
from src.spira_training.shared.core.services.feature_engineering_service import FeatureEngineeringService
from src.spira_training.shared.core.services.randomizer import Randomizer
from src.spira_training.shared.ports.config_loader import ConfigLoader
from src.spira_training.shared.ports.dataset_repository import DatasetRepository
def main():
    config = JsonConfigLoader().load_feature_engineering_config("path/to/config")
    randomizer = Randomizer.initialize_random()
    dataset_repository = DatasetRepository()
    service = FeatureEngineeringService(config, randomizer, dataset_repository)

    service.execute()

if __name__ == "__main__":
    main()