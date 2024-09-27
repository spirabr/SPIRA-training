from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.event import Event


class TrainLogger(ABC):
    @abstractmethod
    def log_event(self, event: Event) -> None: ...
