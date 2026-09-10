import math

import pytest
import torch

from myqwen.config import ModelConfig
from myqwen.modeling import CausalLM
from myqwen.training.profiling import (
    achieved_tflops,
    count_parameters,
    estimate_mfu,
    estimate_training_flops_per_token,
    get_cuda_peak_memory,
)


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        position_embedding_type="rope",
        max_position_embeddings=32,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        mixer_pattern=("attention",),
        tie_word_embeddings=True,
    )


def test_training_flops_per_token_matches_formula():
    model = CausalLM(tiny_config())
    seq_len = 16

    measured = estimate_training_flops_per_token(model, seq_len)
    expected = 6 * count_parameters(model) + 12 * 2 * 4 * 8 * seq_len

    assert measured == expected


def test_achieved_tflops_and_mfu():
    measured_tflops = achieved_tflops(250_000, 120_000_000)
    measured_mfu = estimate_mfu(250_000, 120_000_000, 300.0)

    assert math.isclose(measured_tflops, 30.0)
    assert math.isclose(measured_mfu, 0.1)


def test_invalid_peak_tflops_is_rejected():
    with pytest.raises(ValueError, match="device_peak_tflops"):
        estimate_mfu(100.0, 1000.0, 0.0)


def test_cpu_peak_memory_returns_none():
    assert get_cuda_peak_memory(torch.device("cpu")) is None
