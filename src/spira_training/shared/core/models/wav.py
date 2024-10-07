from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, List, Generic, TypeVar

class Wav(ABC):
    @abstractmethod
    def resize(self, length: int) -> 'Wav':
        pass

    @abstractmethod
    def rescale(self, amplitude: float) -> 'Wav':
        pass

    @abstractmethod
    def combine(self, wav_2: 'Wav') -> 'Wav':
        pass

    @abstractmethod
    def resample(self, actual_sample_rate: int, desired_sample_rate: int) -> 'Wav':
        pass

    @abstractmethod
    def slice(self, start_index: int, end_index: int) -> 'Wav':
        pass

    @abstractmethod
    def concatenate(self, wav: 'Wav') -> 'Wav':
        pass

    @abstractmethod
    def __getattr__(self, name):
        pass

W = TypeVar("W", bound=Wav)

def concatenate_wavs(wavs: List[W]) -> W:
    if not wavs:
        return None

    return reduce(lambda acc, wav: acc.concatenate(wav), wavs)