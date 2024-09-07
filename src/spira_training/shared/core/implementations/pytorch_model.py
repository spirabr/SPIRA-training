from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.dataset import Label
from src.spira_training.shared.core.models.trained_model import TrainedModel


class PytorchModel(TrainedModel):
    def predict(self, feature: Audio) -> Label:
        return Label.NEGATIVE
