from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.audios import Audios
from src.spira_training.shared.core.models.audio import Audio


class AudioProcessor(ABC):
    @abstractmethod
    def get_transformer(self):
        pass

    def process_audio(self, audio: Audio) -> Audio:
        transformer = self.get_transformer()
        feature_wav = transformer(audio.wav)
        transposed_feature_wav = feature_wav.transpose(1, 2)
        reshaped_feature_wav = transposed_feature_wav.reshape(
            transposed_feature_wav.shape[1:]
        )
        return Audio(wav=reshaped_feature_wav, sample_rate=audio.sample_rate)

    def process_audios(self, audios: Audios) -> Audios:
        audio_list = [self.process_audio(audio) for audio in audios]
        return Audios(audios=audio_list, hop_length=audios.hop_length)
