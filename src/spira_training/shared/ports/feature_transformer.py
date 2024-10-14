from abc import ABC, abstractmethod

from src.spira_training.shared.adapters.pytorch_wav import PytorchWav


class FeatureTransformer(ABC):

    @abstractmethod
    def transform(self, wav: PytorchWav) -> PytorchWav:
        pass