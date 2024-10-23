from functools import reduce

import torch
from torchaudio.transforms import Resample
from typing import List


class Wav:
    def __init__(self, tensor):
        self.tensor = tensor

    def resize(self, length: int) -> 'Wav':
        return Wav(self.tensor[0:length])

    def rescale(self, amplitude: float) -> 'Wav':
        return Wav(torch.mul(self.tensor, amplitude / float(self.tensor.max())))

    def combine(self, wav_2: 'Wav') -> 'Wav':
        return Wav(self.tensor + wav_2.tensor)

    def resample(self, actual_sample_rate: int, desired_sample_rate: int) -> 'Wav':
        if desired_sample_rate == actual_sample_rate:
            return self
        resample = Resample(actual_sample_rate, desired_sample_rate)
        return Wav(resample(self.tensor))

    def slice(self, start_index: int, end_index: int) -> 'Wav':
        return Wav(self.tensor[start_index:end_index])

    def concatenate(self, wav: 'Wav') -> 'Wav':
        tensors = [self.tensor, wav.tensor]
        return Wav(torch.cat(tensors, dim=0))

    def __getattr__(self, name):
        return getattr(self.tensor, name)


def concatenate_wavs(wavs: List[Wav]) -> Wav:
    if not wavs:
        return None

    return reduce(lambda acc, wav: acc.concatenate(wav), wavs)


def create_empty_wav() -> Wav:
    return Wav(torch.empty(0))
