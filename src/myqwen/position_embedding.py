import torch
import torch.nn as nn

from myqwen.config import ModelConfig


class AbsolutePositionEmbedding(nn.Module):
    """
    Learned absolute position embedding.

    Input:
        x: [B, T, hidden_size]

    Output:
        x + position_embedding: [B, T, hidden_size]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size

        self.position_embedding = nn.Embedding(
            self.max_position_embeddings,
            self.hidden_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"max_position_embeddings "
                f"{self.max_position_embeddings}"
            )

        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                device=x.device,
            )

            position_ids = position_ids.unsqueeze(0)

        position_embeddings = self.position_embedding(
            position_ids
        )

        return x + position_embeddings




class RotaryEmbedding(nn.Module):
    """
    Text-only 1D Rotary Position Embedding.

    This module only generates cos/sin positional factors.
    It does not directly modify Q/K.

    Stage A:
        head_dim   = 256
        rotary_dim = 64

    So only the first 64 dimensions of each Q/K head
    will receive RoPE.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.rotary_dim = config.rotary_dim
        self.rope_theta = config.rope_theta

        dim_indices = torch.arange(
            0,
            self.rotary_dim,
            2,
            dtype=torch.float32,
        )

        inv_freq = 1.0 / (
            self.rope_theta
            ** (dim_indices / self.rotary_dim)
        )

        # inv_freq is NOT trainable.
        # register_buffer lets it:
        #   1. move with model.cuda()/model.to(device)
        #   2. remain outside model.parameters()
        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        half_dim = x.shape[-1] // 2

        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]

        return torch.cat(
            (-x2, x1),
            dim=-1,
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            position_ids:
                [B, T]
                e.g. [[0, 1, 2, ..., T-1]]

        Returns:
            cos: [B, T, rotary_dim]
            sin: [B, T, rotary_dim]
        """

        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        # [B, T, 1]
        positions = position_ids.to(
            device=self.inv_freq.device,
            dtype=torch.float32,
        ).unsqueeze(-1)

        # [1, 1, rotary_dim / 2]
        inv_freq = self.inv_freq.view(
            1,
            1,
            -1,
        )

        # [B, T, rotary_dim / 2]
        freqs = positions * inv_freq

        emb = torch.cat(
            (freqs, freqs),
            dim=-1,
        )

        cos = emb.cos()
        sin = emb.sin()

        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)

        return cos, sin


    def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Partial RoPE to Q and K.

        Args:
            q:
                [B, num_q_heads, T, head_dim]

            k:
                [B, num_kv_heads, T, head_dim]

            cos / sin:
                [B, T, rotary_dim]

        Returns:
            q_embed:
                same shape as q

            k_embed:
                same shape as k
        """

        # [B, T, rotary_dim]
        # ->
        # [B, 1, T, rotary_dim]
        #
        # The same positional rotation is broadcast
        # across all attention heads.
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        rotary_dim = cos.shape[-1]

        # -------------------------
        # Partial RoPE
        # -------------------------

        q_rot = q[..., :rotary_dim]
        q_pass = q[..., rotary_dim:]

        k_rot = k[..., :rotary_dim]
        k_pass = k[..., rotary_dim:]

        # Standard RoPE:
        #
        # x' = x cos(theta) + R(x) sin(theta)
        #
        q_rot = (
            q_rot * cos
            + self.rotate_half(q_rot) * sin
        )

        k_rot = (
            k_rot * cos
            + self.rotate_half(k_rot) * sin
        )

        # Put the untouched NoPE dimensions back.
        q_embed = torch.cat(
            (q_rot, q_pass),
            dim=-1,
        )

        k_embed = torch.cat(
            (k_rot, k_pass),
            dim=-1,
        )

        return q_embed, k_embed