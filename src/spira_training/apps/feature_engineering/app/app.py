from src.spira_training.shared.core.services.data_processing import load_and_transform_data, load_data


class App:
    def __init__(self, config: Config, randomizer: Randomizer, dataset_repository: DatasetRepository):
        self.config = config
        self.randomizer = randomizer
        self.dataset_repository = dataset_repository

    def execute(self) -> None:
        patients_inputs, controls_inputs, noises = load_data(self.config)
        dataset = generate_dataset(self.config, self.randomizer, patients_inputs, controls_inputs, noises)
        dataset_repository.save_dataset(dataset, self.config.paths.dataset)
