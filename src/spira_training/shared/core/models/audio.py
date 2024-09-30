from src.spira_training.shared.core.models.wav import Wav


class Audio:
    def __init__(self, wav: Wav, sample_rate: int):
        self.wav = wav
        self.sample_rate = sample_rate
