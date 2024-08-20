from spira.config import Config
from spira.utils.randomizer import Randomizer
from src.spira_training.apps.feature_engineering.app import App

def main():
    config = Config.load()
    randomizer = Randomizer()

    app = App(config, randomizer)
    transformed_data = app.execute()
    #save(transformed_data)

if __name__ == "__main__":
    main()