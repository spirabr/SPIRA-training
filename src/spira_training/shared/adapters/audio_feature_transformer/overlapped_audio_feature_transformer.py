from src.spira_training.shared.core.audio_service import create_slices_from_audio, concatenate_audios
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.audio_collection import AudioCollection
from src.spira_training.shared.ports.audio_feature_transformer import AudioFeatureTransformer
from src.spira_training.shared.core.audio_processor import AudioProcessor


class OverlappedAudioFeatureTransformer(AudioFeatureTransformer):
    def __init__(self, audio_processor: AudioProcessor, window_length: int, step_size: int):
        self.audio_processor = audio_processor
        self.window_length = window_length
        self.step_size = step_size

    def transform_into_features(self, audio_collection: AudioCollection) -> AudioCollection:

        return self._overlap_audio_collection(audio_collection)

    def _overlap_audio_collection(self, audio_collection: AudioCollection) -> AudioCollection:
        return AudioCollection(
            [self._overlap_audio(audio) for audio in audio_collection]
        )

    def _overlap_audio(self, audio: Audio) -> Audio:
        audio_slices = create_slices_from_audio(audio, self.window_length, self.step_size)
        processed_audios = self.audio_processor.process_audios(audio_slices)

        return concatenate_audios(processed_audios)