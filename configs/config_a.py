from myqwen.config import ModelConfig


MODEL_CONFIG = ModelConfig(
    # --------------------------------------------------------------
    # Tokenizer
    # GPT-2 tokenizer
    # --------------------------------------------------------------
    vocab_size=50257,

    # --------------------------------------------------------------
    # Backbone
    # --------------------------------------------------------------
    hidden_size=256,
    num_hidden_layers=8,
    intermediate_size=896,       # 3.5 × hidden_size

    # --------------------------------------------------------------
    # Full Attention
    # --------------------------------------------------------------
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=256,

    # --------------------------------------------------------------
    # Gated DeltaNet
    # --------------------------------------------------------------
    linear_num_key_heads=4,
    linear_num_value_heads=4,

    linear_key_head_dim=128,
    linear_value_head_dim=128,

    linear_conv_kernel_dim=4,

    # --------------------------------------------------------------
    # Partial RoPE
    # 256 × 0.25 = 64 dimensions
    # --------------------------------------------------------------
    rope_theta=10_000_000.0,
    partial_rotary_factor=0.25,

    # --------------------------------------------------------------
    # Norm / activation
    # --------------------------------------------------------------
    rms_norm_eps=1e-6,
    hidden_act="silu",

    # --------------------------------------------------------------
    # 3 GDN + 1 Full Attention
    # --------------------------------------------------------------
    full_attention_interval=4,

    # --------------------------------------------------------------
    # Other
    # --------------------------------------------------------------
    attention_dropout=0.0,
    use_bias=False,
    tie_word_embeddings=True,
    initializer_range=0.02,

    max_position_embeddings=512,
)