import math
from typing import cast
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_optimizer import (
    PytorchOptimizer,
)
from typing_extensions import Self

from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_model import (
    PytorchModel,
)
from src.spira_training.shared.core.models.loss import Loss
from src.spira_training.shared.core.models.step import Step
from src.spira_training.shared.core.models.valid_path import ValidPath
import torch


class FilesystemCheckpoint:
    def __init__(self, checkpoint_state: dict):
        self.model_state = checkpoint_state["model"]
        self.optimizer_state = checkpoint_state["optimizer"]
        self.validation_loss = checkpoint_state["validation_loss"]
        self.step = Step(checkpoint_state["step"])

    @classmethod
    def create_initial_checkpoint(
        cls, model: PytorchModel, optimizer: PytorchOptimizer
    ) -> Self:
        return cast(
            Self,
            FilesystemCheckpoint.create(
                model, optimizer, Loss(value=math.inf), Step(0)
            ),
        )

    @classmethod
    def load(cls, checkpoint_path: ValidPath) -> Self:
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
        return cast(Self, FilesystemCheckpoint(checkpoint_state))

    def restore(self, model: PytorchModel, optimizer: PytorchOptimizer) -> Step:
        model.load_state(self.model_state)
        optimizer.load_state(self.optimizer_state)
        return self.step

    @classmethod
    def create(
        cls,
        model: PytorchModel,
        optimizer: PytorchOptimizer,
        validation_loss: Loss,
        step: Step,
    ) -> Self:
        checkpoint_state = {
            "model": model.dump_state(),
            "optimizer": optimizer.dump_state(),
            "validation_loss": validation_loss,
            "step": step,
        }
        return cast(Self, FilesystemCheckpoint(checkpoint_state))

    def save(self, checkpoint_path: ValidPath):
        checkpoint_state = {
            "model": self.model_state,
            "optimizer": self.optimizer_state,
            "validation_loss": self.validation_loss.item(),
            "step": self.step,
        }
        torch.save(checkpoint_state, checkpoint_path)
