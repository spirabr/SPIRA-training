from typing import Iterator

from src.spira_training.shared.core.models.audio import Audio


class AudioCollection:
    def __init__(self, audios: list[Audio], hop_length: int):
        self.audios = audios
        self.hop_length = hop_length

    def __iter__(self) -> Iterator[Audio]:
        return iter(self.audios)