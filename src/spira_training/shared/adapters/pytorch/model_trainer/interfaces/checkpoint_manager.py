from abc import ABC, abstractmethod

from pydantic import BaseModel

from src.spira_training.shared.core.models.loss import Loss
from src.spira_training.shared.core.models.step import Step


class Checkpoint(BaseModel):
    loss: Loss
    step: Step


class CheckpointManager(ABC):
    @abstractmethod
    def update_and_save_checkpoint(self, checkpoint: Checkpoint) -> None: ...
