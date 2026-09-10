from __future__ import annotations

import torch
import torch.nn as nn

from myqwen.config import ModelConfig
from myqwen.modeling.attention.full_attention import KVCache, SoftmaxAttention
from myqwen.modeling.attention.linear_attention import LinearAttention
from myqwen.modeling.layers.mlp import StandardMLP, SwiGLUMLP
from myqwen.modeling.layers.norm import LayerNorm, RMSNorm


def build_norm(config: ModelConfig) -> nn.Module:
    norm_type = config.norm_type.lower()
    if norm_type == "rmsnorm":
        return RMSNorm(config.hidden_size, eps=config.norm_eps)
    if norm_type == "layernorm":
        return LayerNorm(config.hidden_size, eps=config.norm_eps)
    raise ValueError(f"Unsupported norm_type: {config.norm_type}")


def build_mlp(config: ModelConfig) -> nn.Module:
    mlp_type = config.mlp_type.lower()
    if mlp_type == "standard":
        return StandardMLP(config)
    if mlp_type == "swiglu":
        return SwiGLUMLP(config)
    raise ValueError(f"Unsupported mlp_type: {config.mlp_type}")


def build_mixer(config: ModelConfig, layer_type: str) -> nn.Module:
    layer_type = layer_type.lower()
    if layer_type in {"attention", "full_attention"}:
        return SoftmaxAttention(config)
    if layer_type == "linear_attention":
        return LinearAttention(config)
    raise ValueError(f"Unsupported layer_type: {layer_type}")


class DecoderBlock(nn.Module):
    """Pre-Norm decoder block."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.input_norm = build_norm(config)
        self.mixer = build_mixer(config, self.layer_type)
        self.post_mixer_norm = build_norm(config)
        self.mlp = build_mlp(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)

        present_key_value = None

        if self.layer_type in {"attention", "full_attention"}:
            if use_cache:
                hidden_states, present_key_value = self.mixer(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=True,
                )
            else:
                hidden_states = self.mixer(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )
        elif self.layer_type == "linear_attention":
            if use_cache or past_key_value is not None:
                raise NotImplementedError("KV cache is currently implemented only for full softmax attention")
            hidden_states = self.mixer(hidden_states)
        else:
            raise RuntimeError(f"Unsupported layer_type during forward: {self.layer_type}")

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_mixer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if use_cache:
            return hidden_states, present_key_value
        return hidden_states
