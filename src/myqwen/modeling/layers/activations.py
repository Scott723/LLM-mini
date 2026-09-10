# src/myqwen/activations.py

import math

import torch


def relu(x: torch.Tensor) -> torch.Tensor:
    """
    ReLU(x) = max(0, x)
    """
    return torch.maximum(
        x,
        torch.zeros((), dtype=x.dtype, device=x.device),
    )


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU(x) = x * sigmoid(x)
    """
    return x * torch.sigmoid(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Exact GELU:

    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    """
    return 0.5 * x * (
        1.0 + torch.erf(x / math.sqrt(2.0))
    )


def get_activation(name: str):
    name = name.lower()

    if name == "relu":
        return relu

    if name in {"silu", "swish"}:
        return silu

    if name == "gelu":
        return gelu

    raise ValueError(
        f"Unsupported activation: {name}"
    )