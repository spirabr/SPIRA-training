import random
from typing import Any

import numpy as np
import torch

from src.spira_training.shared.ports.random_generator import RandomGenerator


class BaseRandomGenerator(RandomGenerator):
    def __init__(self, seed: int):
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

    def apply_random_seed(self) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

    def get_randint_in_interval(self, first: int, second: int) -> int:
        return random.randint(first, second)

    def get_random_float_in_interval(self, first: float, second: float) -> float:
        return random.uniform(first, second)

    def choose_n_elements(self, elements: list[Any], num_elements: int) -> list[Any]:
        return random.sample(elements, num_elements)

    def get_probability(self, alpha: float, beta: float) -> float:
        return self.random_state.beta(alpha, beta)