import torch
import torch.nn as nn
import torch.nn.functional as F

from myqwen.config import ModelConfig
from myqwen.modeling.attention.utils import repeat_kv


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """
    Positive feature map for classic linear attention.

    phi(x) = ELU(x) + 1
    """
    return F.elu(x) + 1.0


def causal_linear_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Reference causal kernel linear attention.

    Intentionally uses a Python loop so the recurrent-state update
    is explicit and easy to study.

    Args:
        q, k, v: [B, H, T, D]

    Returns:
        output: [B, H, T, D]

    Recurrence:
        S_t = S_{t-1} + phi(k_t) v_t^T
        Z_t = Z_{t-1} + phi(k_t)

        o_t = phi(q_t)^T S_t / (phi(q_t)^T Z_t + eps)
    """
    q = elu_feature_map(q)
    k = elu_feature_map(k)

    batch_size, num_heads, seq_len, head_dim = q.shape

    # S_t: [B,H,D,D]
    kv_state = torch.zeros(
        batch_size,
        num_heads,
        head_dim,
        head_dim,
        device=q.device,
        dtype=q.dtype,
    )

    # Z_t: [B,H,D]
    k_state = torch.zeros(
        batch_size,
        num_heads,
        head_dim,
        device=q.device,
        dtype=q.dtype,
    )

    outputs = []

    for t in range(seq_len):
        q_t = q[:, :, t, :]
        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]

        # [B,H,D,1] * [B,H,1,D] -> [B,H,D,D]
        kv_state = kv_state + (
            k_t.unsqueeze(-1)
            * v_t.unsqueeze(-2)
        )

        k_state = k_state + k_t

        # q_t^T S_t -> [B,H,D]
        numerator = torch.matmul(
            q_t.unsqueeze(-2),
            kv_state,
        ).squeeze(-2)

        # q_t^T Z_t -> [B,H,1]
        denominator = (
            q_t * k_state
        ).sum(
            dim=-1,
            keepdim=True,
        )

        output_t = numerator / (denominator + eps)
        outputs.append(output_t)

    return torch.stack(
        outputs,
        dim=2,
    )


class LinearAttention(nn.Module):
    """
    Classic causal kernel linear attention baseline.

    This is not Gated DeltaNet. It is a simple educational linear
    attention implementation whose sequence-length complexity is
    linear in T, while the recurrent state still contains D x D work.
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

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, hidden_size]

        Returns:
            output: [B, T, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.head_dim,
        )
        k = self.k_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        v = self.v_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )

        # [B,T,H,D] -> [B,H,T,D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Reuse the same MHA/GQA/MQA head-sharing logic.
        k = repeat_kv(
            k,
            self.num_key_value_groups,
        )
        v = repeat_kv(
            v,
            self.num_key_value_groups,
        )

        output = causal_linear_attention_reference(
            q,
            k,
            v,
        )

        # [B,H,T,D] -> [B,T,H*D]
        output = (
            output
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.q_dim,
            )
        )

        return self.o_proj(output)
