from src.spira_training.shared.core.models.audio import Audio


class GeneratedAudioCollection:
    def __init__(self, generated_audios: list['Audio']):
        self.generated_audios = generated_audios

    def __len__(self) -> int:
        return len(self.generated_audios)

    def copy_using(self, audios: list['Audio']) -> 'GeneratedAudioCollection':
        return GeneratedAudioCollection(audios)