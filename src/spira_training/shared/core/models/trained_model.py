from abc import ABC, abstractmethod


class TrainedModel(ABC):
    @abstractmethod
    def dump_state(self) -> dict: ...

    @abstractmethod
    def load_state(self, state_dict: dict): ...
