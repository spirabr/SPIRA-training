from src.spira_training.shared.adapters.random_generators.random_generator_test import RandomGeneratorTest
from src.spira_training.shared.adapters.random_generators.random_generator_train import RandomGeneratorTrain
from src.spira_training.shared.core.models.operation_mode import OperationMode
from src.spira_training.shared.ports.random_generator import RandomGenerator


class RandomGeneratorFactory:
    def create_random_generator(self, seed: int, operation_mode: OperationMode) -> RandomGenerator:
        if operation_mode == OperationMode.TRAIN:
            return RandomGeneratorTrain(seed)
        elif operation_mode == OperationMode.TEST:
            return RandomGeneratorTest(seed)
        else:
            raise ValueError("You must configure the operation mode to train or test")
