from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.models.pytorch_audio import (
    create_empty_wav,
    PytorchAudio,
)
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.pytorch_audio_factory import (
    PytorchAudioFactory,
)


class FakePytorchAudioFactory(PytorchAudioFactory):
    def __init__(self) -> None:
        self.call_args = []

    def create_pytorch_from_audio(self, audio: Audio) -> PytorchAudio:
        self.call_args.append(audio)
        return PytorchAudio(wav=create_empty_wav(), sample_rate=0)

    def assert_called_with(self, audio: Audio):
        assert audio in self.call_args, f"Expected {audio} to be in {self.call_args}"
