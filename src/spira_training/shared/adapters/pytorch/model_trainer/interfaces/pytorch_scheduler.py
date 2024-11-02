from abc import ABC, abstractmethod


class PytorchScheduler(ABC):
    @abstractmethod
    def step(self) -> None:
        pass
