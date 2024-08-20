from src.spira_training.shared.core.services.data_processing import load_and_transform_data


class App:
    def __init__(self, config: Config, randomizer: Randomizer):
        self.config = config
        self.randomizer = randomizer

    def execute(self):
        transformed_data = load_and_transform_data(self.config, self.randomizer)
        return transformed_data
