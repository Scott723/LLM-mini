import torch

from myqwen.config import ModelConfig
from myqwen.generation import GenerationConfig, generate
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


def test_greedy_generation_cache_matches_no_cache():
    torch.manual_seed(0)
    model = CausalLM(tiny_config()).eval()
    prompt = torch.tensor([[1, 2, 3, 4]])

    without_cache = generate(
        model,
        prompt,
        GenerationConfig(max_new_tokens=6, do_sample=False, use_cache=False),
    )
    with_cache = generate(
        model,
        prompt,
        GenerationConfig(max_new_tokens=6, do_sample=False, use_cache=True),
    )

    assert torch.equal(with_cache, without_cache)


def test_generation_preserves_model_training_mode():
    torch.manual_seed(0)
    model = CausalLM(tiny_config())
    model.train()
    prompt = torch.tensor([[1, 2, 3]])

    generate(model, prompt, GenerationConfig(max_new_tokens=2, use_cache=True))

    assert model.training is True


def test_generation_rejects_context_overflow():
    model = CausalLM(tiny_config()).eval()
    prompt = torch.ones((1, 30), dtype=torch.long)

    try:
        generate(model, prompt, GenerationConfig(max_new_tokens=3))
    except ValueError as error:
        assert "exceeds max_position_embeddings" in str(error)
    else:
        raise AssertionError("Expected generation context overflow to raise ValueError")
