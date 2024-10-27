from src.spira_training.shared.adapters.random_generators.random_generator_factory import RandomGeneratorFactory
from src.spira_training.shared.core.models.operation_mode import OperationMode
from src.spira_training.shared.adapters.random_generators.random_generator_test import RandomGeneratorTest
from src.spira_training.shared.adapters.random_generators.random_generator_train import RandomGeneratorTrain
import pytest

def test_creates_train_random_generator():
    factory = RandomGeneratorFactory()
    generator = factory.create_random_generator(seed=42, operation_mode=OperationMode.TRAIN)
    assert isinstance(generator, RandomGeneratorTrain)

def test_creates_test_random_generator():
    factory = RandomGeneratorFactory()
    generator = factory.create_random_generator(seed=42, operation_mode=OperationMode.TEST)
    assert isinstance(generator, RandomGeneratorTest)

def test_raises_value_error_for_invalid_operation_mode():
    factory = RandomGeneratorFactory()
    with pytest.raises(ValueError, match="You must configure the operation mode to train or test"):
        factory.create_random_generator(seed=42, operation_mode=None)