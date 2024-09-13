from typing import List, Set
from src.spira_training.shared.core.models.model_params import ModelParams
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.dataset import Label
from src.spira_training.shared.core.models.trained_model import TrainedModel


class FakeModel(TrainedModel):
    def __init__(self) -> None:
        self._predicted = set()

    def predict(self, feature: Audio) -> Label:
        self._predicted.add(feature)
        return Label.NEGATIVE

    def predict_batch(self, features_batch: List[Audio]) -> List[Label]:
        labels = []
        for feature in features_batch:
            self._predicted.add(feature)
            labels.append(Label.NEGATIVE)
        return labels

    def dump_state(self) -> dict:
        return dict()

    def load_state(self, state_dict: dict):
        pass

    def get_parameters(self) -> list[ModelParams]:
        return list()

    def assert_predicted_once(self, feature: Audio):
        # return feature in self._predicted
        assert (
            feature in self._predicted
        ), f"Feature {self._feature_str(feature)} not in set {self._feature_set_str(self._predicted)}"

    def _feature_str(self, feature: Audio) -> str:
        return str(hash(feature))

    def _feature_set_str(self, feature_set: Set[Audio]):
        result = "["
        for feature in feature_set:
            result += self._feature_str(feature) + ", "
        result += "]"

        return result
