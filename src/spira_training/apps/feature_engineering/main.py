from src.spira_training.apps.feature_engineering.app.app import App
from src.spira_training.shared.core.services.randomizer import Randomizer
from src.spira_training.shared.ports.config_loader import ConfigLoader
from src.spira_training.shared.ports.dataset_repository import DatasetRepository


def main():
    config = ConfigLoader.load()
    randomizer = Randomizer.initialize_random()
    dataset_repository = DatasetRepository()

    app = App(config, randomizer, dataset_repository)
    app.execute()

if __name__ == "__main__":
    main()