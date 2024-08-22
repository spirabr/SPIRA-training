from spira.config import Config
from spira.utils.randomizer import Randomizer
from src.spira_training.apps.feature_engineering.app import App

def main():
    config = Config.load()
    randomizer = Randomizer()
    dataset_repository = DatasetRepository()

    app = App(config, randomizer, dataset_repository)
    app.execute()

if __name__ == "__main__":
    main()