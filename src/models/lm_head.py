import torch
from torch import nn
from transformers.activations import gelu


class MixedTabLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.float_predictor = nn.Linear(config.hidden_size, 1)

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        float_predictions = self.float_predictor(x).squeeze(-1)
        x = self.decoder(x)

        return x, float_predictions
