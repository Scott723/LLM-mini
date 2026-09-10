# src/myqwen/mlp.py

import torch
import torch.nn as nn

from myqwen.config import ModelConfig
from myqwen.modeling.layers.activations import get_activation


class SwiGLUMLP(nn.Module):
    """
    SwiGLU feed-forward network.

    Input:
        x: [B, T, hidden_size]

    Output:
        y: [B, T, hidden_size]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=config.mlp_bias
        )

        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=config.mlp_bias
        )

        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=config.mlp_bias
        )

        self.activation = get_activation(
            config.hidden_act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.activation(self.gate_proj(x))

        up = self.up_proj(x)

        hidden = gate * up

        output = self.down_proj(hidden)

        return output

class StandardMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias
        )

        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias
        )

        self.activation = get_activation(
            config.hidden_act
        )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)

        return x
    