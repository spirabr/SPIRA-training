from src.spira_training.shared.core.interfaces.random import Random
from src.spira_training.shared.core.models.config import Config
from src.spira_training.shared.core.services.data_processing import load_data, generate_dataset
from src.spira_training.shared.ports import dataset_repository
from src.spira_training.shared.ports.dataset_repository import DatasetRepository


class App:
    def __init__(self, config: Config, randomizer: Random, dataset_repository: DatasetRepository):
        self.config = config
        self.randomizer = randomizer
        self.dataset_repository = dataset_repository

    def execute(self) -> None:
        patients_inputs, controls_inputs, noises = load_data(self.config)
        dataset = generate_dataset(self.config, self.randomizer, patients_inputs, controls_inputs, noises)
        dataset_repository.save_dataset(dataset, self.config.paths.dataset)
