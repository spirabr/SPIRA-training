from typing import Optional

from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_checkpoint_manager import (
    PytorchCheckpointManager,
)

from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_checkpoint_manager.filesystem_checkpoint_builder import (
    FileSystemCheckpointBuilder,
)

from spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_checkpoint_manager.filesystem_checkpoint import (
    FilesystemCheckpoint,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_optimizer import (
    PytorchOptimizer,
)

from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_model import (
    PytorchModel,
)
from src.spira_training.shared.core.models.loss import Loss
from src.spira_training.shared.core.models.step import Step


class FilesystemPytorchCheckpointManager(PytorchCheckpointManager):
    def __init__(
        self,
        checkpoint_builder: FileSystemCheckpointBuilder,
        model: PytorchModel,
        optimizer: PytorchOptimizer,
        initial_checkpoint: Optional[FilesystemCheckpoint],
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
        self, previous_checkpoint: Optional[FilesystemCheckpoint]
    ) -> FilesystemCheckpoint:
        return (
            previous_checkpoint
            if previous_checkpoint
            else FilesystemCheckpoint.create_initial_checkpoint(
                self.model, self.optimizer
            )
        )

    def _create_last_checkpoint(self, loss: Loss, step: Step) -> FilesystemCheckpoint:
        current_checkpoint = self.checkpoint_builder.create_checkpoint(
            self.model, self.optimizer, loss, step
        )
        return current_checkpoint if current_checkpoint else self.last_checkpoint

    def _define_best_checkpoint(self) -> FilesystemCheckpoint:
        if self.last_checkpoint.validation_loss < self.best_checkpoint.validation_loss:
            return self.last_checkpoint
        return self.best_checkpoint
