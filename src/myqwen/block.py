import torch
import torch.nn as nn

from myqwen.config import ModelConfig
from myqwen.full_attention import SoftmaxAttention
from myqwen.linear_attention import LinearAttention
from myqwen.mlp import StandardMLP, SwiGLUMLP
from myqwen.norm import LayerNorm, RMSNorm


def build_norm(config: ModelConfig) -> nn.Module:
    norm_type = config.norm_type.lower()

    if norm_type == "rmsnorm":
        return RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
        )

    if norm_type == "layernorm":
        return LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
        )

    raise ValueError(
        f"Unsupported norm_type: {config.norm_type}"
    )


def build_mlp(config: ModelConfig) -> nn.Module:
    mlp_type = config.mlp_type.lower()

    if mlp_type == "standard":
        return StandardMLP(config)

    if mlp_type == "swiglu":
        return SwiGLUMLP(config)

    raise ValueError(
        f"Unsupported mlp_type: {config.mlp_type}"
    )


def build_mixer(
    config: ModelConfig,
    layer_type: str,
) -> nn.Module:
    layer_type = layer_type.lower()

    if layer_type in {
        "attention",
        "full_attention",
    }:
        return SoftmaxAttention(config)

    if layer_type == "linear_attention":
        return LinearAttention(config)

    raise ValueError(
        f"Unsupported layer_type: {layer_type}"
    )


class DecoderBlock(nn.Module):
    """
    Pre-Norm decoder block.

    h = x + Mixer(Norm(x))
    y = h + MLP(Norm(h))

    The mixer can be selected independently for every layer
    through config.layer_types.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        self.input_norm = build_norm(config)

        self.mixer = build_mixer(
            config,
            self.layer_type,
        )

        self.post_mixer_norm = build_norm(config)

        self.mlp = build_mlp(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:
                [B, T, hidden_size]

            position_ids:
                [B, T] or [T].
                Used by RoPE-based full attention.

            attention_mask:
                Optional [B, T] padding mask.
                Currently used by full softmax attention.

        Returns:
            hidden_states:
                [B, T, hidden_size]
        """

        # ----------------------------------------------------------
        # Token mixer sub-layer
        #
        # h = x + Mixer(Norm(x))
        # ----------------------------------------------------------
        residual = hidden_states

        hidden_states = self.input_norm(
            hidden_states
        )

        if self.layer_type in {
            "attention",
            "full_attention",
        }:
            hidden_states = self.mixer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        elif self.layer_type == "linear_attention":
            hidden_states = self.mixer(
                hidden_states
            )

        else:
            raise RuntimeError(
                f"Unsupported layer_type during forward: "
                f"{self.layer_type}"
            )

        hidden_states = residual + hidden_states

        # ----------------------------------------------------------
        # Feed-forward sub-layer
        #
        # y = h + MLP(Norm(h))
        # ----------------------------------------------------------
        residual = hidden_states

        hidden_states = self.post_mixer_norm(
            hidden_states
        )

        hidden_states = self.mlp(
            hidden_states
        )

        hidden_states = residual + hidden_states

        return hidden_states