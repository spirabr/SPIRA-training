import math
import os
from typing import Optional, cast
from typing_extensions import Self

from src.spira_training.shared.ports.path_validator import PathValidator
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_optimizer_wrapper import (
    PytorchOptimizerWrapper,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_model import (
    PytorchModel,
)
from src.spira_training.shared.core.models.loss import Loss
from src.spira_training.shared.core.models.step import Step
from src.spira_training.shared.core.models.valid_path import ValidPath
import torch


class Checkpoint:
    def __init__(self, checkpoint_state: dict):
        self.model_state = checkpoint_state["model"]
        self.optimizer_state = checkpoint_state["optimizer"]
        self.validation_loss = checkpoint_state["validation_loss"]
        self.step = Step(checkpoint_state["step"])

    @classmethod
    def create_initial_checkpoint(
        cls, model: PytorchModel, optimizer: PytorchOptimizerWrapper
    ) -> Self:
        return cast(
            Self, Checkpoint.create(model, optimizer, Loss(value=math.inf), Step(0))
        )

    @classmethod
    def load(cls, checkpoint_path: ValidPath) -> Self:
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
        return cast(Self, Checkpoint(checkpoint_state))

    def restore(self, model: PytorchModel, optimizer: PytorchOptimizerWrapper) -> Step:
        model.load_state(self.model_state)
        optimizer.load_state(self.optimizer_state)
        return self.step

    @classmethod
    def create(
        cls,
        model: PytorchModel,
        optimizer: PytorchOptimizerWrapper,
        validation_loss: Loss,
        step: Step,
    ) -> Self:
        checkpoint_state = {
            "model": model.dump_state(),
            "optimizer": optimizer.dump_state(),
            "validation_loss": validation_loss,
            "step": step,
        }
        return cast(Self, Checkpoint(checkpoint_state))

    def save(self, checkpoint_path: ValidPath):
        checkpoint_state = {
            "model": self.model_state,
            "optimizer": self.optimizer_state,
            "validation_loss": self.validation_loss.item(),
            "step": self.step,
        }
        torch.save(checkpoint_state, checkpoint_path)


class FileSystemCheckpointBuilder:
    def __init__(
        self,
        checkpoint_dir: ValidPath,
        checkpoint_interval: int,
        fs_path_validator: PathValidator,
    ):
        if checkpoint_interval <= 1:
            raise ValueError("Checkpoint interval should be greater than one")

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.fs_path_validator = fs_path_validator

    def create_checkpoint(
        self,
        model: PytorchModel,
        optimizer: PytorchOptimizerWrapper,
        loss: Loss,
        step: Step,
    ) -> Optional[Checkpoint]:
        return (
            Checkpoint.create(model, optimizer, loss, step)
            if self._should_checkpoint(step)
            else None
        )

    def save_checkpoint_with_step(self, checkpoint: Checkpoint):
        checkpoint_path = self.fs_path_validator.validate_path(
            os.path.join(self.checkpoint_dir, f"checkpoint_{checkpoint.step}.pt")
        )
        checkpoint.save(checkpoint_path)

    def save_checkpoint_with_prefix(self, checkpoint: Checkpoint, prefix: str):
        checkpoint_path = self.fs_path_validator.validate_path(
            os.path.join(self.checkpoint_dir, f"{prefix}_checkpoint.pt")
        )
        checkpoint.save(checkpoint_path)

    def _should_checkpoint(self, step: Step) -> bool:
        return int(step) % self.checkpoint_interval == 0


class PytorchCheckpointManager:
    def __init__(
        self,
        checkpoint_builder: FileSystemCheckpointBuilder,
        model: PytorchModel,
        optimizer: PytorchOptimizerWrapper,
        initial_checkpoint: Optional[Checkpoint],
    ):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_builder = checkpoint_builder

        self.last_checkpoint = self._initialize_checkpoint(initial_checkpoint)
        self.best_checkpoint = self.last_checkpoint

    def update_and_save_checkpoints(self, loss: Loss, step: Step):
        self._update_checkpoints(loss, step)
        self._save_checkpoints()

    def _update_checkpoints(self, loss: Loss, step: Step):
        self.last_checkpoint = self._create_last_checkpoint(loss, step)
        self.best_checkpoint = self._define_best_checkpoint()

    def _save_checkpoints(self):
        self.checkpoint_builder.save_checkpoint_with_step(self.last_checkpoint)
        self.checkpoint_builder.save_checkpoint_with_prefix(
            self.best_checkpoint, "best"
        )

    def _initialize_checkpoint(
        self, previous_checkpoint: Optional[Checkpoint]
    ) -> Checkpoint:
        return (
            previous_checkpoint
            if previous_checkpoint
            else Checkpoint.create_initial_checkpoint(self.model, self.optimizer)
        )

    def _create_last_checkpoint(self, loss: Loss, step: Step) -> Checkpoint:
        current_checkpoint = self.checkpoint_builder.create_checkpoint(
            self.model, self.optimizer, loss, step
        )
        return current_checkpoint if current_checkpoint else self.last_checkpoint

    def _define_best_checkpoint(self) -> Checkpoint:
        if self.last_checkpoint.validation_loss < self.best_checkpoint.validation_loss:
            return self.last_checkpoint
        return self.best_checkpoint
