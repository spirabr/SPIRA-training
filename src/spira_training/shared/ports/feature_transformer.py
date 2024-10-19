from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.wav import Wav


class FeatureTransformer(ABC):

    @abstractmethod
    def transform(self, wav: Wav) -> Wav:
        pass