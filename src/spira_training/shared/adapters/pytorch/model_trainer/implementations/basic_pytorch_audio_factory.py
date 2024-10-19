from src.spira_training.shared.adapters.pytorch.models.pytorch_tensor import (
    PytorchTensor,
)
from src.spira_training.shared.core.models.audio import Audio
import torch
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_audio_factory import (
    PytorchTensorFactory,
)


class BasicPytorchTensorFactory(PytorchTensorFactory):
    def create_tensor_from_audio(self, audio: Audio) -> PytorchTensor:
        return PytorchTensor(torch.tensor(audio.wav.tensor, dtype=torch.float32))
