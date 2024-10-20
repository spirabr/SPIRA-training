from typing import List

from src.spira_training.shared.adapters.pytorch.models.pytorch_parameter import (
    PytorchParameter,
)

from src.spira_training.shared.adapters.pytorch.models.pytorch_label import (
    PytorchLabel,
)

from src.spira_training.shared.adapters.pytorch.models.pytorch_tensor import (
    PytorchTensor,
)
import torch

from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_model import (
    PytorchModel,
)


class FakePytorchModel(PytorchModel):
    def __init__(self) -> None:
        self._predicted = {}

    def predict(self, feature: PytorchTensor) -> PytorchLabel:
        if self._predicted.get(feature):
            self._predicted[feature] += 1
        else:
            self._predicted[feature] = 1

        return make_false_label()

    def predict_batch(self, features_batch: List[PytorchTensor]) -> List[PytorchLabel]:
        labels = []
        for feature in features_batch:
            label = self.predict(feature)
            labels.append(label)
        return labels

    def dump_state(self) -> dict:
        return dict()

    def load_state(self, state_dict: dict):
        pass

    def get_parameters(self) -> list[PytorchParameter]:
        return []

    def assert_predicted_once(self, feature: PytorchTensor):
        self.assert_predicted_times(feature=feature, times=1)

    def assert_predicted_times(self, feature: PytorchTensor, times: int):
        feature_times = self._predicted.get(feature, 0)
        assert (
            times <= 0 or feature_times > 0
        ), f"Feature {self._feature_str(feature)} not in set {self._feature_set_str(self._predicted)}"

        assert (
            feature_times == times
        ), f"Feature {self._feature_str(feature)} expected to be trained {times} times, but was {feature_times}"

    def _feature_str(self, feature: PytorchTensor) -> str:
        return str(hash(feature))

    def _feature_set_str(self, feature_set: dict):
        result = "["
        for feature in feature_set:
            result += self._feature_str(feature) + ", "
        result += "]"

        return result


def make_false_label() -> PytorchLabel:
    return PytorchLabel(torch.empty(1))
