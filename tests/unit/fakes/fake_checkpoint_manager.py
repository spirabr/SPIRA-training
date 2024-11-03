from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_checkpoint_manager import (
    Checkpoint,
    PytorchCheckpointManager,
)


class FakeCheckpointManager(PytorchCheckpointManager):
    def __init__(self) -> None:
        self.checkpoints: list[Checkpoint] = []

    def update_and_save_checkpoint(self, checkpoint: Checkpoint):
        self.checkpoints.append(checkpoint)
