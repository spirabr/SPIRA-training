import random
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import torch
from typing_extensions import Self

from spira.core.domain.enum import OperationMode


class Random(ABC):
    def __init__(self, seed: int):
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

    @abstractmethod
    def create_random(self, seed) -> Self:
        pass

    def apply_random_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def get_randint_in_interval(first: int, second: int) -> int:
        return random.randint(first, second)  # type: ignore

    @staticmethod
    def get_random_float_in_interval(first: float, second: float) -> float:
        return random.uniform(first, second)  # type: ignore

    @staticmethod
    def choose_n_elements(elements: list[Any], num_elements: int) -> list[Any]:
        return random.sample(elements, num_elements)  # type: ignore

    def get_probability(self, alpha: float, beta: float) -> float:
        return self.random_state.beta(alpha, beta)


class TrainRandom(Random):
    def create_random(self, seed) -> Random:  # type: ignore
        return cast(Random, self)


class TestRandom(Random):
    def create_random(self, seed) -> Random:  # type: ignore
        new_seed = self.seed * seed
        return cast(Random, TestRandom(seed=new_seed))


def initialize_random(config, operation_mode) -> Random:
    match operation_mode:
        case OperationMode.TRAIN:
            return TrainRandom(config.seed)
        case OperationMode.TEST:
            return TestRandom(config.seed)
        case _:
            raise RuntimeError("You must configure the operation mode to train or test")
