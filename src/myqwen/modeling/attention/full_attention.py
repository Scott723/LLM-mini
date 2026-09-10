from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from myqwen.config import ModelConfig
from myqwen.modeling.attention.utils import repeat_kv
from myqwen.modeling.layers.norm import RMSNorm
from myqwen.modeling.layers.position_embedding import RotaryEmbedding


KVCache = tuple[torch.Tensor, torch.Tensor]
VALID_ATTENTION_BACKENDS = {"eager", "sdpa"}


def _build_causal_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a lower-right-aligned causal mask for q_len <= kv_len."""
    query_len = q.shape[-2]
    key_len = k.shape[-2]

    if key_len < query_len:
        raise ValueError(f"key_len must be >= query_len, got key_len={key_len}, query_len={query_len}")

    past_length = key_len - query_len
    query_positions = torch.arange(query_len, device=q.device) + past_length
    key_positions = torch.arange(key_len, device=q.device)
    causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

    if attention_mask is None:
        return causal_mask

    if attention_mask.ndim != 2 or attention_mask.shape != (q.shape[0], key_len):
        raise ValueError(
            f"attention_mask must have shape {(q.shape[0], key_len)}, got {tuple(attention_mask.shape)}"
        )

    key_mask = attention_mask.to(dtype=torch.bool, device=q.device)[:, None, None, :]
    return causal_mask[None, None, :, :] & key_mask


def scaled_dot_product_attention_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference causal attention that explicitly materializes attention weights."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    causal_mask = _build_causal_mask(q, k, attention_mask)
    scores = scores.masked_fill(~causal_mask, float("-inf"))

    attn_weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)

    if dropout_p > 0.0:
        attn_weights = torch.dropout(attn_weights, dropout_p, training)

    return torch.matmul(attn_weights, v), attn_weights


def scaled_dot_product_attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    PyTorch SDPA backend.

    Square, unpadded training attention uses is_causal=True, allowing
    PyTorch to select an optimized fused kernel. KV-cache decoding and
    padding use an explicit lower-right-aligned boolean mask because
    q_len can be smaller than kv_len.
    """
    query_len = q.shape[-2]
    key_len = k.shape[-2]
    effective_dropout = dropout_p if training else 0.0

    if query_len == key_len and attention_mask is None:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=effective_dropout,
            is_causal=True,
        )

    causal_mask = _build_causal_mask(q, k, attention_mask)

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=causal_mask,
        dropout_p=effective_dropout,
        is_causal=False,
    )


class SoftmaxAttention(nn.Module):
    """General causal self-attention with eager/SDPA backends and KV cache."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_dim = self.num_attention_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.attention_dropout = config.attention_dropout
        self.qk_norm_enabled = config.qk_norm
        self.output_gate_enabled = config.attention_output_gate
        self.attention_backend = "eager"

        self.q_proj = nn.Linear(self.hidden_size, self.q_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.q_dim, self.hidden_size, bias=config.attention_bias)

        if self.qk_norm_enabled:
            self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.rotary_emb = RotaryEmbedding(config) if config.position_embedding_type == "rope" else None
        self.gate_proj = (
            nn.Linear(self.hidden_size, self.q_dim, bias=config.attention_bias)
            if self.output_gate_enabled
            else None
        )

    def set_attention_backend(self, backend: str) -> None:
        backend = backend.lower()

        if backend not in VALID_ATTENTION_BACKENDS:
            raise ValueError(
                f"attention backend must be one of {sorted(VALID_ATTENTION_BACKENDS)}, got {backend!r}"
            )

        self.attention_backend = backend

    def _validate_cache(self, past_key_value: KVCache, batch_size: int) -> None:
        past_key, past_value = past_key_value
        expected_prefix = (batch_size, self.num_key_value_heads)

        if past_key.ndim != 4 or past_value.ndim != 4:
            raise ValueError("KV cache tensors must have shape [B, num_kv_heads, T, head_dim]")
        if past_key.shape[:2] != expected_prefix or past_value.shape[:2] != expected_prefix:
            raise ValueError(
                f"KV cache must start with shape {expected_prefix}, got K={tuple(past_key.shape)}, "
                f"V={tuple(past_value.shape)}"
            )
        if past_key.shape[-1] != self.head_dim or past_value.shape[-1] != self.head_dim:
            raise ValueError(f"KV cache head_dim must be {self.head_dim}")
        if past_key.shape[-2] != past_value.shape[-2]:
            raise ValueError("Cached K and V must have the same sequence length")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_attn_weights: bool = False,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ):
        if past_key_value is not None and not use_cache:
            raise ValueError("past_key_value requires use_cache=True")

        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

            cos, sin = self.rotary_emb(position_ids, dtype=q.dtype)
            q, k = self.rotary_emb.apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            self._validate_cache(past_key_value, batch_size)
            past_key, past_value = past_key_value
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present_key_value = (k, v) if use_cache else None

        repeated_k = repeat_kv(k, self.num_key_value_groups)
        repeated_v = repeat_kv(v, self.num_key_value_groups)

        if self.attention_backend == "sdpa" and not return_attn_weights:
            attn_output = scaled_dot_product_attention_sdpa(
                q,
                repeated_k,
                repeated_v,
                attention_mask=attention_mask,
                dropout_p=self.attention_dropout,
                training=self.training,
            )
            attn_weights = None
        else:
            attn_output, attn_weights = scaled_dot_product_attention_eager(
                q,
                repeated_k,
                repeated_v,
                attention_mask=attention_mask,
                dropout_p=self.attention_dropout,
                training=self.training,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.q_dim)

        if self.gate_proj is not None:
            attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states))

        output = self.o_proj(attn_output)

        if use_cache and return_attn_weights:
            return output, attn_weights, present_key_value
        if use_cache:
            return output, present_key_value
        if return_attn_weights:
            return output, attn_weights
        return output
