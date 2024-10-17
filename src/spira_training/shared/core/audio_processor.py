from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_audio_factory import (
    PytorchAudioFactory,
)
from src.spira_training.shared.core.models.audio_collection import AudioCollection
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.ports.feature_transformer import FeatureTransformer


class AudioProcessor:
    def __init__(
        self,
        feature_transformer: FeatureTransformer,
        pytorch_audio_factory: PytorchAudioFactory,
    ):
        self.feature_transformer = feature_transformer
        self.pytorch_audio_factory = pytorch_audio_factory

    def process_audio(self, audio: Audio) -> Audio:
        pytorch_audio = self.pytorch_audio_factory.create_pytorch_from_audio(audio)
        feature_wav = self.feature_transformer.transform(pytorch_audio.wav)
        transposed_feature_wav = feature_wav.transpose(1, 2)
        reshaped_feature_wav = transposed_feature_wav.reshape(
            transposed_feature_wav.shape[1:]
        )
        return Audio(wav=reshaped_feature_wav, sample_rate=audio.sample_rate)

    def process_audios(self, audios: AudioCollection) -> AudioCollection:
        audio_list = [self.process_audio(audio) for audio in audios]
        return AudioCollection(audios=audio_list, hop_length=audios.hop_length)
