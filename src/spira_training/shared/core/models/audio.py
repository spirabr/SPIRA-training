from src.spira_training.shared.core.models.wav import Wav


class Audio:
    def __init__(self, wav: Wav = None, sample_rate: int = 0):
        self.wav = wav
        self.sample_rate = sample_rate
