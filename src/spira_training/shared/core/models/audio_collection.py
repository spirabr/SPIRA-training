from typing import Iterator, Optional

from src.spira_training.shared.core.models.audio import Audio


class AudioCollection:
    def __init__(self, audio_list: list[Audio], hop_length: int):
        self.audio_list = audio_list
        self.hop_length = hop_length
        self._min_audio_length: Optional[int] = None
        self._max_audio_length: Optional[int] = None

    def __iter__(self) -> Iterator[Audio]:
        return iter(self.audio_list)

    def __len__(self) -> int:
        return len(self.audio_list)


    def copy_using(self, audio_list: list[Audio]) -> 'AudioCollection':
        return AudioCollection(audio_list, self.hop_length)

    def get_max_audio_length(self) -> int:
        if self._max_audio_length is None:
            self._calculate_min_max_audio_length()
        return self._max_audio_length

    def get_min_audio_length(self) -> int:
        if self._min_audio_length is None:
            self._calculate_min_max_audio_length()
        return self._min_audio_length

    def _calculate_min_max_audio_length(self):
        audio_lengths = [self._calculate_audio_length(audio) for audio in self.audio_list]
        self._min_audio_length = min(audio_lengths)
        self._max_audio_length = max(audio_lengths)

    def _calculate_audio_length(self, audio: Audio) -> int:
        return int((audio.wav.shape[1] / self.hop_length) + 1)