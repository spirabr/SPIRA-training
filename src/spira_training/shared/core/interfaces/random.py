from abc import ABC, abstractmethod
import numpy as np

class Random(ABC):
    def __init__(self, seed: int):
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

    @abstractmethod
    def initialize_random(self, seed):
        pass
