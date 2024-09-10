from abc import ABC, abstractmethod
from typing import Generic, List

from src.spira_training.shared.core.models.model_params import ModelParamsT
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.dataset import Label


class TrainedModel(ABC, Generic[ModelParamsT]):
    @abstractmethod
    def predict(self, feature: Audio) -> Label: ...

    @abstractmethod
    def predict_batch(self, features_batch: List[Audio]) -> List[Label]: ...

    @abstractmethod
    def dump_state(self) -> dict: ...

    @abstractmethod
    def load_state(self, state_dict: dict): ...

    @abstractmethod
    def get_parameters(self) -> list[ModelParamsT]: ...
