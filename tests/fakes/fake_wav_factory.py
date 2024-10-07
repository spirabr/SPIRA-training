from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.wav import (
    Wav,
    create_empty_wav,
)
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.wav_factory import (
    WavFactory,
)


class FakeWavFactory(WavFactory):
    def __init__(self) -> None:
        self.call_args = []

    def create_wav_from_audio(self, audio: Audio) -> Wav:
        self.call_args.append(audio)
        return create_empty_wav()

    def assert_called_with(self, audio: Audio):
        assert audio in self.call_args, f"Expected {audio} to be in {self.call_args}"
