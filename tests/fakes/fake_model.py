from typing import List

import torch

from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.pytorch_model import (
    PytorchModel,
    PytorchLabel,
    PytorchWav,
)


class FakePytorchModel(PytorchModel):
    def __init__(self) -> None:
        self._predicted = {}

    def predict(self, feature: PytorchWav) -> PytorchLabel:
        if self._predicted.get(feature):
            self._predicted[feature] += 1
        else:
            self._predicted[feature] = 1

        return make_false_label()

    def predict_batch(self, features_batch: List[PytorchWav]) -> List[PytorchLabel]:
        labels = []
        for feature in features_batch:
            label = self.predict(feature)
            labels.append(label)
        return labels

    def dump_state(self) -> dict:
        return dict()

    def load_state(self, state_dict: dict):
        pass

    def assert_predicted_once(self, feature: PytorchWav):
        self.assert_predicted_times(feature=feature, times=1)

    def assert_predicted_times(self, feature: PytorchWav, times: int):
        feature_times = self._predicted.get(feature, 0)
        assert (
            times <= 0 or feature_times > 0
        ), f"Feature {self._feature_str(feature)} not in set {self._feature_set_str(self._predicted)}"

        assert (
            feature_times == times
        ), f"Feature {self._feature_str(feature)} expected to be trained {times} times, but was {feature_times}"

    def _feature_str(self, feature: PytorchWav) -> str:
        return str(hash(feature))

    def _feature_set_str(self, feature_set: dict):
        result = "["
        for feature in feature_set:
            result += self._feature_str(feature) + ", "
        result += "]"

        return result


def make_false_label() -> PytorchLabel:
    return PytorchLabel(torch.empty(1))
