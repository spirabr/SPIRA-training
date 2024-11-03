from typing import List
from src.spira_training.shared.core.models.event import Event
from src.spira_training.shared.ports.train_logger import TrainLogger


class FakeTrainLogger(TrainLogger):
    def __init__(self) -> None:
        self.logged_events: List[Event] = []

    def log_event(self, event: Event) -> None:
        self.logged_events.append(event)
