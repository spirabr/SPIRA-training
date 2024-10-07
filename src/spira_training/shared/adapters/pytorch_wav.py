import torch
from torchaudio.transforms import Resample
from typing import List

from src.spira_training.shared.core.models.wav import Wav


class PytorchWav(Wav):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def resize(self, length: int) -> 'PytorchWav':
        return PytorchWav(self.data[0:length])

    def rescale(self, amplitude: float) -> 'PytorchWav':
        return PytorchWav(torch.mul(self.data, amplitude / float(self.data.max())))

    def combine(self, wav_2: 'PytorchWav') -> 'PytorchWav':
        return PytorchWav(self.data + wav_2.data)

    def resample(self, actual_sample_rate: int, desired_sample_rate: int) -> 'PytorchWav':
        if desired_sample_rate == actual_sample_rate:
            return self
        resample = Resample(actual_sample_rate, desired_sample_rate)
        return PytorchWav(resample(self.data))

    def slice(self, start_index: int, end_index: int) -> 'PytorchWav':
        return PytorchWav(self.data[start_index:end_index])

    def concatenate(self, wav: 'PytorchWav') -> 'PytorchWav':
        tensors = [self.data, wav.data]
        return PytorchWav(torch.cat(tensors, dim=0))

    def __getattr__(self, name):
        return getattr(self.data, name)