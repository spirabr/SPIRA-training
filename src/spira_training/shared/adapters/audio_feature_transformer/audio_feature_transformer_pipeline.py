from src.spira_training.shared.core.models.audio_collection import AudioCollection
from src.spira_training.shared.ports.audio_feature_transformer import AudioFeatureTransformer

class AudioFeatureTransformerPipeline(AudioFeatureTransformer):
    def __init__(self, transformers: list[AudioFeatureTransformer]):
        self.transformers = transformers

    def transform_into_features(self, audios: AudioCollection) -> AudioCollection:
        for transformer in self.transformers:
            audios = transformer.transform_into_features(audios)
        return audios