from src.spira_training.shared.core.audio_processor import AudioProcessor
from src.spira_training.shared.core.audio_service import add_padding_to_audio_collection
from src.spira_training.shared.core.models.audio_collection import AudioCollection
from src.spira_training.shared.ports.audio_feature_transformer import AudioFeatureTransformer


class PaddedAudioFeatureTransformer(AudioFeatureTransformer):
    def __init__(self, audio_processor: AudioProcessor):
        self.audio_processor = audio_processor

    def transform_into_features(self, audios: AudioCollection) -> AudioCollection:
        processed_audios = self.audio_processor.process_audios(audios)

        return add_padding_to_audio_collection(processed_audios)
