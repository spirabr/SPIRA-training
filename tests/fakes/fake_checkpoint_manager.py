from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.checkpoint_manager import (
    Checkpoint,
    CheckpointManager,
)


class FakeCheckpointManager(CheckpointManager):
    def __init__(self) -> None:
        self.checkpoints: list[Checkpoint] = []

    def update_and_save_checkpoint(self, checkpoint: Checkpoint):
        self.checkpoints.append(checkpoint)
