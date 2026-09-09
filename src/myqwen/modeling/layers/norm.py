import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Function:
        y = gamma * x / sqrt(mean(x^2) + eps)

    Input:
        x: [..., hidden_size]

    Output:
        same shape as x

    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps

        # Learnable scale parameter gamma
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        # Compute normalization statistics in FP32 for numerical stability.
        x = x.float()

        variance = x.pow(2).mean(dim=-1, keepdim=True)  # 平方求和平均

        x = x * torch.rsqrt(variance + self.eps)  # 1/sqrt(variance + eps)

        # Cast back to the original dtype, e.g. BF16.
        x = x.to(input_dtype)

        return self.weight * x


class LayerNorm(nn.Module):
    """
    Layer Normalization.

    Function:
        y = gamma * (x - mean) / sqrt(var + eps) + beta
        
    Input:
        x: [..., hidden_size]

    Output:
        same shape as x

    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        # Use FP32 to compute normalization statistics.
        x = x.float()

        mean = x.mean(dim=-1, keepdim=True)

        variance = (
            (x - mean)
            .pow(2)
            .mean(dim=-1, keepdim=True)
        )

        x = (x - mean) * torch.rsqrt(
            variance + self.eps
        )

        x = x.to(input_dtype)

        return self.weight * x + self.bias