from src.spira_training.apps.feature_engineering.configs.feature_engineering_config import FeatureEngineeringConfig
from src.spira_training.shared.core.interfaces.random import Random
from src.spira_training.shared.ports import dataset_repository
from src.spira_training.shared.ports.dataset_repository import DatasetRepository

class FeatureEngineeringService:
    def __init__(self, config: FeatureEngineeringConfig, randomizer: Random, dataset_repository: DatasetRepository):
        self.config = config
        self.randomizer = randomizer
        self.dataset_repository = dataset_repository

    def execute(self) -> None:
        patients_inputs, controls_inputs, noises = self._load_data(self.config)
        dataset = self._generate_dataset(self.config, self.randomizer, patients_inputs, controls_inputs, noises)
        dataset_repository.save_dataset(dataset, self.config.paths.dataset)

    def _load_data(self, config: FeatureEngineeringConfig):
        pass

    def _generate_dataset(self):
        pass

