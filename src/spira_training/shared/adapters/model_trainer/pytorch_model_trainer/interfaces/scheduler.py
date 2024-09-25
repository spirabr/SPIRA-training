from abc import ABC, abstractmethod


class Scheduler(ABC):
    @abstractmethod
    def step(self) -> None:
        pass
