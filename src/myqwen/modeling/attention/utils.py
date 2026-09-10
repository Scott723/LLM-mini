import torch


def repeat_kv(
    x: torch.Tensor,
    n_rep: int,
) -> torch.Tensor:
    """
    Repeat KV heads so that their head count matches Q heads.

    Input:
        x: [B, H_kv, T, D]

    Output:
        [B, H_q, T, D]
    """

    if n_rep == 1:
        return x

    batch_size, num_kv_heads, seq_len, head_dim = x.shape

    x = x[:, :, None, :, :]

    x = x.expand(
        batch_size,
        num_kv_heads,
        n_rep,
        seq_len,
        head_dim,
    )

    return x.reshape(
        batch_size,
        num_kv_heads * n_rep,
        seq_len,
        head_dim,
    )