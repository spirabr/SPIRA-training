from typing import List
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.ports.audios_repository import AudiosRepository


class FakeAudiosRepository(AudiosRepository):
    def __init__(self):
        self.audios = {}
        self.get_audio_called = False

    def get_audios(self, path: str) -> List[Audio]:
        return self.audios.get(path, [])

    def get_audio(self, path: str) -> Audio:
        self.get_audio_called = True
        return self.audios.get(path, Audio())


def make_audios():
    return [make_audio() for i in range(0, 3)]


def make_audio():
    return Audio()
