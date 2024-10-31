import torch
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_cnn_builder import (
    PytorchCnnBuilder,
    PytorchCnnConfig,
)
import torch.nn as nn


class PaddedFeaturesPytorchCnnBuilder(PytorchCnnBuilder):
    def __init__(
        self, config: PytorchCnnConfig, num_features: int, max_audio_length: int
    ):
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
