import math

from src.spira_training.shared.core.models.wav import Wav


class Audio:
    def __init__(self, wav: Wav = None, sample_rate: int = 0):
        self.wav = wav
        self.sample_rate = sample_rate

    def __len__(self):
        return math.ceil(len(self.wav) / self.sample_rate)

    def add_padding(self, max_audio_length):
        return Audio(wav=self.wav.add_padding(max_audio_length), sample_rate=self.sample_rate)