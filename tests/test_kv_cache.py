import torch

from myqwen.config import ModelConfig
from myqwen.modeling import CausalLM


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


def test_incremental_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    model = CausalLM(tiny_config()).eval()
    input_ids = torch.randint(0, 64, (2, 6))

    with torch.no_grad():
        full_logits = model(input_ids)

        past_key_values = None
        incremental_logits = []

        for position in range(input_ids.shape[1]):
            token = input_ids[:, position : position + 1]
            logits, past_key_values = model(
                token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            incremental_logits.append(logits)

    cached_logits = torch.cat(incremental_logits, dim=1)
    torch.testing.assert_close(cached_logits, full_logits, atol=1e-5, rtol=1e-4)


def test_gqa_cache_keeps_unrepeated_kv_heads():
    torch.manual_seed(0)
    config = tiny_config()
    model = CausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    with torch.no_grad():
        _, past_key_values = model(input_ids, use_cache=True)

    assert len(past_key_values) == config.num_hidden_layers

    for key, value in past_key_values:
        expected_shape = (2, config.num_key_value_heads, 5, config.head_dim)
        assert key.shape == expected_shape
        assert value.shape == expected_shape
