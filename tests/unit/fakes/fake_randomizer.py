from src.spira_training.shared.core.interfaces.random import Random

class FakeRandomizer(Random):
    def __init__(self):
        self.seed = None

    def initialize_random(self, seed: int) -> Random:
        self.seed = seed
        return self