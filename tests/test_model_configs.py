import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str):
    namespace = runpy.run_path(str(PROJECT_ROOT / "configs" / "model" / name))
    return namespace["MODEL_CONFIG"]


def analytical_parameter_count(config) -> int:
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim

    embedding = config.vocab_size * config.hidden_size
    attention = (
        config.hidden_size * q_dim
        + config.hidden_size * kv_dim
        + config.hidden_size * kv_dim
        + q_dim * config.hidden_size
    )
    mlp = 3 * config.hidden_size * config.intermediate_size
    block_norms = 2 * config.hidden_size
    final_norm = config.hidden_size

    return embedding + config.num_hidden_layers * (attention + mlp + block_norms) + final_norm


def test_scaling_model_dimensions_and_parameter_counts():
    model_19m = load_config("model_19m.py")
    model_300m = load_config("model_300m.py")
    model_800m = load_config("model_800m.py")

    assert analytical_parameter_count(model_19m) == 19_161_600
    assert analytical_parameter_count(model_300m) == 303_163_392
    assert analytical_parameter_count(model_800m) == 813_376_512

    for config in (model_19m, model_300m, model_800m):
        assert config.head_dim == 64
        assert config.intermediate_size == 3 * config.hidden_size
        assert config.num_attention_heads == 2 * config.num_key_value_heads
        assert config.num_attention_heads * config.head_dim == config.hidden_size
