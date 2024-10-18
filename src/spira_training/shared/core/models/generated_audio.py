from src.spira_training.shared.core.models.wav import Wav


class GeneratedAudio:
    def __init__(self, wav: Wav):
        self.wav = wav