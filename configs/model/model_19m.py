from myqwen.config import ModelConfig


MODEL_CONFIG = ModelConfig(
    vocab_size=50257,

    hidden_size=256,
    num_hidden_layers=8,
    intermediate_size=768,

    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,

    qk_norm=False,
    attention_output_gate=False,
    attention_bias=False,
    attention_dropout=0.0,

    position_embedding_type="rope",
    rope_theta=10_000.0,
    partial_rotary_factor=1.0,
    max_position_embeddings=512,

    norm_type="rmsnorm",
    norm_eps=1e-6,

    mlp_type="swiglu",
    hidden_act="silu",
    mlp_bias=False,

    mixer_pattern=("attention",),

    tie_word_embeddings=True,
    initializer_range=0.02,
)
