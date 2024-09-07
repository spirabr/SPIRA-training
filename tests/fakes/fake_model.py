from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.dataset import Label
from src.spira_training.shared.core.models.trained_model import TrainedModel


class FakeModel(TrainedModel):
    def __init__(self) -> None:
        self._predicted = set()

    def predict(self, feature: Audio) -> Label:
        self._predicted.add(feature)
        return Label.NEGATIVE

    def assert_predicted_once(self, feature: Audio):
        # return feature in self._predicted
        assert (
            feature in self._predicted
        ), f"Feature {feature} not in set {self._predicted}"
