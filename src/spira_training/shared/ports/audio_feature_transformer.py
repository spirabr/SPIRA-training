from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.audio_collection import AudioCollection


class AudioFeatureTransformer(ABC):
    @abstractmethod
    def transform_into_features(self, audios: AudioCollection) -> AudioCollection:
        pass
