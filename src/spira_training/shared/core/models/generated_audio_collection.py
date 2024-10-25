from src.spira_training.shared.core.models.audio import Audio


class GeneratedAudioCollection:
    def __init__(self, generated_audio_list: list['Audio']):
        self.generated_audio_list = generated_audio_list

    def __len__(self) -> int:
        return len(self.generated_audio_list)

    def copy_using(self, audio_list: list['Audio']) -> 'GeneratedAudioCollection':
        return GeneratedAudioCollection(audio_list)