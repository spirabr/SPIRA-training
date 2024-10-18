from src.spira_training.shared.core.models.generated_audio import GeneratedAudio


class GeneratedAudioCollection:
    def __init__(self, generated_audios: list[GeneratedAudio]):
        self.generated_audios = generated_audios
