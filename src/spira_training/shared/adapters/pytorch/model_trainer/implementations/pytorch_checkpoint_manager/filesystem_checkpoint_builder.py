import os
from typing import Optional

from spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_checkpoint_manager.filesystem_checkpoint import (
    FilesystemCheckpoint,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_optimizer import (
    PytorchOptimizer,
)

from src.spira_training.shared.ports.path_validator import PathValidator
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_model import (
    PytorchModel,
)
from src.spira_training.shared.core.models.loss import Loss
from src.spira_training.shared.core.models.step import Step
from src.spira_training.shared.core.models.valid_path import ValidPath


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
        optimizer: PytorchOptimizer,
        loss: Loss,
        step: Step,
    ) -> Optional[FilesystemCheckpoint]:
        return (
            FilesystemCheckpoint.create(model, optimizer, loss, step)
            if self._should_checkpoint(step)
            else None
        )

    def save_checkpoint_with_step(self, checkpoint: FilesystemCheckpoint):
        checkpoint_path = self.fs_path_validator.validate_path(
            os.path.join(self.checkpoint_dir, f"checkpoint_{checkpoint.step}.pt")
        )
        checkpoint.save(checkpoint_path)

    def save_checkpoint_with_prefix(
        self, checkpoint: FilesystemCheckpoint, prefix: str
    ):
        checkpoint_path = self.fs_path_validator.validate_path(
            os.path.join(self.checkpoint_dir, f"{prefix}_checkpoint.pt")
        )
        checkpoint.save(checkpoint_path)

    def _should_checkpoint(self, step: Step) -> bool:
        return int(step) % self.checkpoint_interval == 0
