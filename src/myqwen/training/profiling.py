from __future__ import annotations

from dataclasses import dataclass

import torch


GIB = 1024 ** 3


@dataclass(frozen=True)
class CudaPeakMemory:
    """Cumulative CUDA peak-memory statistics since the last reset."""

    allocated_gib: float
    reserved_gib: float


def count_parameters(model: torch.nn.Module) -> int:
    """Count unique parameters exposed by model.parameters()."""
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_training_flops_per_token(model: torch.nn.Module, seq_len: int) -> float:
    """
    Estimate decoder-only training FLOPs per input token.

    Common nanoGPT/PaLM-style approximation:

        FLOPs/token ~= 6N + 12 * L * H * D_head * T

    N is parameter count. The second term captures the quadratic
    attention matmuls that parameter count alone does not represent.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("MFU estimation requires model.config")

    unsupported = sorted(
        {
            layer_type
            for layer_type in config.layer_types
            if layer_type not in {"attention", "full_attention"}
        }
    )
    if unsupported:
        raise NotImplementedError(
            "MFU estimation currently supports only full softmax attention; "
            f"unsupported layer types: {unsupported}"
        )

    parameter_flops = 6.0 * count_parameters(model)
    attention_flops = (
        12.0
        * config.num_hidden_layers
        * config.num_attention_heads
        * config.head_dim
        * seq_len
    )
    return parameter_flops + attention_flops


def achieved_tflops(tokens_per_second: float, flops_per_token: float) -> float:
    if tokens_per_second < 0:
        raise ValueError("tokens_per_second must be non-negative")
    if flops_per_token <= 0:
        raise ValueError("flops_per_token must be positive")
    return tokens_per_second * flops_per_token / 1e12


def estimate_mfu(
    tokens_per_second: float,
    flops_per_token: float,
    device_peak_tflops: float,
) -> float:
    """Return MFU as a ratio, for example 0.35 means 35%."""
    if device_peak_tflops <= 0:
        raise ValueError("device_peak_tflops must be positive")
    return achieved_tflops(tokens_per_second, flops_per_token) / device_peak_tflops


def reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_cuda_peak_memory(device: torch.device) -> CudaPeakMemory | None:
    if device.type != "cuda":
        return None

    return CudaPeakMemory(
        allocated_gib=torch.cuda.max_memory_allocated(device) / GIB,
        reserved_gib=torch.cuda.max_memory_reserved(device) / GIB,
    )
