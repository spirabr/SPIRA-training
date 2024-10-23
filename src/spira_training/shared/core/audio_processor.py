from src.spira_training.shared.core.models.audio_collection import AudioCollection
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.generated_audio_collection import GeneratedAudioCollection
from src.spira_training.shared.ports.feature_transformer import FeatureTransformer


class AudioProcessor:
    def __init__(
        self,
        feature_transformer: FeatureTransformer,
    ):
        self.feature_transformer = feature_transformer

    def process_audio(self, audio: Audio) -> Audio:
        feature_wav = self.feature_transformer.transform(audio.wav)
        transposed_feature_wav = feature_wav.transpose(1, 2)
        reshaped_feature_wav = transposed_feature_wav.reshape(
            transposed_feature_wav.shape[1:]
        )
        return Audio(wav=reshaped_feature_wav, sample_rate=audio.sample_rate)

    def process_audios(self, audios: AudioCollection | GeneratedAudioCollection) -> AudioCollection | GeneratedAudioCollection:
        audio_list = [self.process_audio(audio) for audio in audios]
        return audios.copy_using(audios=audio_list)
