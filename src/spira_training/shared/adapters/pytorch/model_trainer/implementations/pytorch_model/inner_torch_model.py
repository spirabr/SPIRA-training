from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_model.mish import (
    Mish,
)

from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_cnn_builder import (
    PytorchCnnBuilder,
)
import torch
import torch.nn as nn


class InnerTorchModel(nn.Module):
    def __init__(self, cnn_builder: PytorchCnnBuilder):
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
