import torch
import torch.nn as nn

from myqwen.config import ModelConfig
from myqwen.modeling.layers.norm import RMSNorm
from myqwen.modeling.attention.utils import repeat_kv
from myqwen.modeling.layers.position_embedding import RotaryEmbedding


def scaled_dot_product_attention_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation of causal scaled dot-product attention.

    Args:
        q, k, v: [B, H, T, D]
        attention_mask:
            Optional padding mask [B, T].
            True / 1 = valid token, False / 0 = padding token.

    Returns:
        output: [B, H, T, D]
        attn_weights: [B, H, T, T]
    """
    head_dim = q.shape[-1]
    scale = head_dim ** -0.5

    # [B,H,T,D] @ [B,H,D,T] -> [B,H,T,T]
    scores = torch.matmul(
        q,
        k.transpose(-2, -1),
    )
    scores = scores * scale

    seq_len = q.shape[-2]

    # Causal mask: token t can only attend to positions <= t.
    causal_mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            device=q.device,
            dtype=torch.bool,
        )
    )
    scores = scores.masked_fill(
        ~causal_mask,
        float("-inf"),
    )

    # Optional key padding mask.
    if attention_mask is not None:
        key_mask = attention_mask.to(
            dtype=torch.bool,
            device=q.device,
        )
        key_mask = key_mask[:, None, None, :]  # [B,1,1,T]
        scores = scores.masked_fill(
            ~key_mask,
            float("-inf"),
        )

    # Softmax in FP32 for numerical stability.
    attn_weights = torch.softmax(
        scores.float(),
        dim=-1,
    ).to(q.dtype)

    if dropout_p > 0.0:
        attn_weights = torch.dropout(
            attn_weights,
            dropout_p,
            training,
        )

    # [B,H,T,T] @ [B,H,T,D] -> [B,H,T,D]
    output = torch.matmul(
        attn_weights,
        v,
    )

    return output, attn_weights


class SoftmaxAttention(nn.Module):
    """
    General causal self-attention.

    Supported through config:
        - MHA / GQA / MQA
        - optional QK RMSNorm
        - optional RoPE / Partial RoPE
        - optional output gate
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_dim = self.num_attention_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )

        self.attention_dropout = config.attention_dropout
        self.qk_norm_enabled = config.qk_norm
        self.output_gate_enabled = config.attention_output_gate

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.q_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.kv_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.kv_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.q_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        if self.qk_norm_enabled:
            self.q_norm = RMSNorm(
                self.head_dim,
                eps=config.norm_eps,
            )
            self.k_norm = RMSNorm(
                self.head_dim,
                eps=config.norm_eps,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        if config.position_embedding_type == "rope":
            self.rotary_emb = RotaryEmbedding(config)
        else:
            self.rotary_emb = None

        if self.output_gate_enabled:
            self.gate_proj = nn.Linear(
                self.hidden_size,
                self.q_dim,
                bias=config.attention_bias,
            )
        else:
            self.gate_proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_attn_weights: bool = False,
    ):
        """
        Args:
            hidden_states: [B, T, hidden_size]
            position_ids: [B, T] or [T], only needed by RoPE
            attention_mask: optional [B, T] padding mask
            return_attn_weights: whether to also return [B,H,T,T]

        Returns:
            output: [B, T, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Q / K / V projections.
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Split into heads.
        q = q.view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.head_dim,
        )
        k = k.view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        v = v.view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )

        # Optional per-head QK Norm.
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # [B,T,H,D] -> [B,H,T,D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Optional Full / Partial RoPE.
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(
                    seq_len,
                    device=hidden_states.device,
                ).unsqueeze(0)

            cos, sin = self.rotary_emb(
                position_ids,
                dtype=q.dtype,
            )
            q, k = self.rotary_emb.apply_rotary_pos_emb(
                q,
                k,
                cos,
                sin,
            )

        # MHA: n_rep = 1; GQA/MQA: logically repeat KV heads.
        k = repeat_kv(
            k,
            self.num_key_value_groups,
        )
        v = repeat_kv(
            v,
            self.num_key_value_groups,
        )

        attn_output, attn_weights = scaled_dot_product_attention_eager(
            q,
            k,
            v,
            attention_mask=attention_mask,
            dropout_p=self.attention_dropout,
            training=self.training,
        )

        # [B,H,T,D] -> [B,T,H*D]
        attn_output = (
            attn_output
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.q_dim,
            )
        )

        # Optional output gate, e.g. Qwen3.5-like preset.
        if self.gate_proj is not None:
            gate = self.gate_proj(hidden_states)
            attn_output = (
                attn_output
                * torch.sigmoid(gate)
            )

        output = self.o_proj(attn_output)

        if return_attn_weights:
            return output, attn_weights

        return output

