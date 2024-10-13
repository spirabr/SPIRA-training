from abc import ABC, abstractmethod

from pydantic import BaseModel

from src.spira_training.shared.core.models.loss import Loss


class Checkpoint(BaseModel):
    loss: Loss
    step: int


class CheckpointManager(ABC):
    @abstractmethod
    def update_and_save_checkpoint(self, checkpoint: Checkpoint) -> None: ...
