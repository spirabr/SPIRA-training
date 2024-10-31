from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_model.inner_torch_model import (
    InnerTorchModel,
)
from src.spira_training.shared.adapters.pytorch.models.pytorch_parameter import (
    PytorchParameter,
)
from src.spira_training.shared.adapters.pytorch.models.pytorch_tensor import (
    PytorchTensor,
)
from src.spira_training.shared.adapters.pytorch.models.pytorch_label import (
    PytorchLabel,
)

from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_model import (
    PytorchModel,
)


import torch


class SimplePytorchModel(PytorchModel):
    def __init__(self, model: InnerTorchModel):
        self._inner_model = model

    def dump_state(self) -> dict:
        return self._inner_model.state_dict()

    def load_state(self, state_dict: dict):
        self._inner_model.load_state_dict(state_dict)

    def predict(self, feature: PytorchTensor) -> PytorchLabel:
        return self._inner_model(feature)

    def predict_batch(self, features_batch: list[PytorchTensor]) -> list[PytorchLabel]:
        return self._inner_model(torch.tensor(features_batch))

    def get_parameters(self) -> list[PytorchParameter]:
        return [
            PytorchParameter(parameter) for parameter in self._inner_model.parameters()
        ]
