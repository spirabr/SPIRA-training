from src.spira_training.shared.core.models.audio_collection import AudioCollection
from src.spira_training.shared.ports.audio_feature_transformer import AudioFeatureTransformer

class AudioFeatureTransformerPipeline(AudioFeatureTransformer):
    def __init__(self, transformers: list[AudioFeatureTransformer]):
        self.transformers = transformers

    def transform_into_features(self, audio_collection: AudioCollection) -> AudioCollection:
        for transformer in self.transformers:
            audio_collection = transformer.transform_into_features(audio_collection)
        return audio_collection