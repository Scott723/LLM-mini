from myqwen.config import ModelConfig


MODEL_CONFIG = ModelConfig(

    # ==============================================================
    # Tokenizer
    # GPT-2 tokenizer
    # ==============================================================

    vocab_size=50257,

    # ==============================================================
    # Backbone
    # ==============================================================

    hidden_size=256,
    num_hidden_layers=8,

    # SwiGLU FFN
    intermediate_size=768,

    # ==============================================================
    # Attention
    #
    # Q:
    #   4 heads × 64 = 256
    #
    # KV:
    #   2 heads × 64 = 128
    #
    # => GQA
    # ==============================================================

    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,

    qk_norm=False,
    attention_output_gate=False,

    attention_bias=False,
    attention_dropout=0.0,

    # ==============================================================
    # Position embedding
    #
    # Full RoPE:
    #   64 × 1.0 = 64 rotary dimensions
    # ==============================================================

    position_embedding_type="rope",

    rope_theta=10_000.0,
    partial_rotary_factor=1.0,

    max_position_embeddings=512,

    # ==============================================================
    # Normalization
    # ==============================================================

    norm_type="rmsnorm",
    norm_eps=1e-6,

    # ==============================================================
    # MLP
    # ==============================================================

    mlp_type="swiglu",
    hidden_act="silu",
    mlp_bias=False,

    # ==============================================================
    # Mixer layout
    #
    # Stage A uses a standard Transformer:
    # every layer is full attention.
    # ==============================================================

    mixer_pattern=(
        "attention",
    ),

    # ==============================================================
    # Embedding / initialization
    # ==============================================================

    tie_word_embeddings=True,
    initializer_range=0.02,
)