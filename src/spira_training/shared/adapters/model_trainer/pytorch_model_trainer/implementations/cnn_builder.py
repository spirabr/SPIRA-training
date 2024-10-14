import torch
from src.spira_training.shared.adapters.model_trainer.pytorch_model_trainer.interfaces.cnn_builder import (
    CnnBuilder,
    CnnConfig,
)
import torch.nn as nn


class PaddedFeaturesCnnBuilder(CnnBuilder):
    def __init__(self, config: CnnConfig, num_features: int, max_audio_length: int):
        super().__init__(config, num_features)
        self.max_audio_length = max_audio_length

    def build_fc1(self, conv: torch.nn.modules.container.Sequential) -> nn.Linear:
        # it's very useful because if you change the convolutional architecture the model calculate its, and you don't need change this :)
        # I prefer activate the network in toy example because is easier than calculate the conv output
        # get zeros input
        inp = torch.zeros(1, 1, self.max_audio_length, self.num_features)
        # get out shape
        toy_activation_shape = conv(inp).shape
        # set fully connected input dim
        fc1_input_dim = (
            toy_activation_shape[1] * toy_activation_shape[2] * toy_activation_shape[3]
        )
        return nn.Linear(fc1_input_dim, self.config.fc1_dim)

    def reshape_x(self, x):
        # x: [B, T*n_filters*num_feature]
        return x.view(x.size(0), -1)


class OverlappedFeaturesCNNBuilder(CnnBuilder):
    def __init__(self, config: CnnConfig, num_features: int):
        super().__init__(config, num_features)

    def build_fc1(self, conv: torch.nn.modules.container.Sequential) -> nn.Linear:
        # dynamic calculation num_feature, it's useful if you use max-pooling or other pooling in feature dim, and this model don't break
        inp = torch.zeros(1, 1, 500, self.num_features)
        # get out shape
        return nn.Linear(4 * conv(inp).shape[-1], self.config.fc1_dim)

    def reshape_x(self, x):
        # x: [B, T, n_filters*num_feature]
        return x.view(x.size(0), x.size(1), -1)
