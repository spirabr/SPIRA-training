from typing import List
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.ports.audios_repository import AudiosRepository


class FakeAudiosRepository(AudiosRepository):
    def __init__(self):
        self.audios = {}

    def get_audios(self, path: str) -> List[Audio]:
        return self.audios.get(path, [])


def make_audios():
    return [Audio(), Audio(), Audio()]
