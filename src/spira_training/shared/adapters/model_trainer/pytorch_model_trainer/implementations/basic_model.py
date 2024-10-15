from src.spira_training.shared.adapters.pytorch.models.pytorch_parameter import (
    PytorchParameter,
)
from src.spira_training.shared.adapters.pytorch.models.pytorch_wav import (
    PytorchWav,
)
from src.spira_training.shared.adapters.pytorch.models.pytorch_label import (
    PytorchLabel,
)

from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.pytorch_model import (
    PytorchModel,
)

from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.implementations.mish import (
    Mish,
)

from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.cnn_builder import (
    CnnBuilder,
)
import torch
import torch.nn as nn


class InnerTorchModel(nn.Module):
    def __init__(self, cnn_builder: CnnBuilder):
        super().__init__()

        self.conv = self._build_cnn()
        self.mish = Mish()

        self.fc1 = cnn_builder.build_fc1(self.conv)
        self.fc2 = cnn_builder.build_fc2()
        self.dropout = cnn_builder.define_dropout()
        self.reshape_x = cnn_builder.reshape_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, num_feature]
        x = x.unsqueeze(1)
        # x: [B, 1, T, num_feature]
        x = self.conv(x)
        # x: [B, n_filters, T, num_feature]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, n_filters, num_feature]
        x = self.reshape_x(x)  # type: ignore
        # x: [B, T, fc2_dim]
        x = self.fc1(x)
        x = self.mish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        y = torch.sigmoid(x)
        return y

    def _build_cnn(self) -> nn.Sequential:
        layers = [
            # cnn1
            nn.Conv2d(1, 32, kernel_size=(7, 1), dilation=(2, 1)),
            nn.GroupNorm(16, 32),
            self.mish,
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(p=0.7),
            # cnn2
            nn.Conv2d(32, 16, kernel_size=(5, 1), dilation=(2, 1)),
            nn.GroupNorm(8, 16),  # Normalizacao
            self.mish,  # suavizacao da camada anterior
            nn.MaxPool2d(kernel_size=(2, 1)),  # pooling
            nn.Dropout(p=0.7),  # activation function
            # cnn3
            nn.Conv2d(16, 8, kernel_size=(3, 1), dilation=(2, 1)),
            nn.GroupNorm(4, 8),
            self.mish,
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(p=0.7),
            # cnn4
            nn.Conv2d(8, 4, kernel_size=(2, 1), dilation=(1, 1)),
            nn.GroupNorm(2, 4),
            self.mish,
            nn.Dropout(p=0.7),
        ]
        return nn.Sequential(*layers)


class BasicModel(PytorchModel):
    def __init__(self, model: InnerTorchModel):
        self._inner_model = model

    def dump_state(self) -> dict:
        return self._inner_model.state_dict()

    def load_state(self, state_dict: dict):
        self._inner_model.load_state_dict(state_dict)

    def predict(self, feature: PytorchWav) -> PytorchLabel:
        return self._inner_model(feature)

    def predict_batch(self, features_batch: list[PytorchWav]) -> list[PytorchLabel]:
        return self._inner_model(torch.tensor(features_batch))

    def get_parameters(self) -> list[PytorchParameter]:
        return [
            PytorchParameter(parameter) for parameter in self._inner_model.parameters()
        ]
