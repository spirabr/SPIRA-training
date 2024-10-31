import torch
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_cnn_builder import (
    PytorchCnnBuilder,
    PytorchCnnConfig,
)
import torch.nn as nn


class OverlappedFeaturesPytorchCnnBuilder(PytorchCnnBuilder):
    def __init__(self, config: PytorchCnnConfig, num_features: int):
        super().__init__(config, num_features)

    def build_fc1(self, conv: torch.nn.modules.container.Sequential) -> nn.Linear:
        # dynamic calculation num_feature, it's useful if you use max-pooling or other pooling in feature dim, and this model don't break
        inp = torch.zeros(1, 1, 500, self.num_features)
        # get out shape
        return nn.Linear(4 * conv(inp).shape[-1], self.config.fc1_dim)

    def reshape_x(self, x):
        # x: [B, T, n_filters*num_feature]
        return x.view(x.size(0), x.size(1), -1)
