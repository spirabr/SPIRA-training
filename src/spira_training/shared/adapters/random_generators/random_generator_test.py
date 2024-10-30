from src.spira_training.shared.adapters.random_generators.base_random_generator import BaseRandomGenerator


class RandomGeneratorTest(BaseRandomGenerator):
    def __init__(self, seed: int):
        super().__init__(seed=seed * seed)