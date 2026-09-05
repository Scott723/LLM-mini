from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Configuration of the MyQwen3.5 text backbone.

    This config only describes model architecture.
    Training hyperparameters such as learning rate, batch size,
    and training steps should not be placed here.
    """

    # ------------------------------------------------------------------
    # Vocabulary / basic model dimensions
    # ------------------------------------------------------------------
    vocab_size: int

    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int

    # ------------------------------------------------------------------
    # Full Attention
    # ------------------------------------------------------------------
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    # ------------------------------------------------------------------
    # Gated DeltaNet
    # ------------------------------------------------------------------
    linear_num_key_heads: int
    linear_num_value_heads: int

    linear_key_head_dim: int
    linear_value_head_dim: int

    linear_conv_kernel_dim: int

    # ------------------------------------------------------------------
    # RoPE
    # ------------------------------------------------------------------
    rope_theta: float
    partial_rotary_factor: float

    # ------------------------------------------------------------------
    # Normalization / activation
    # ------------------------------------------------------------------
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    # ------------------------------------------------------------------
    # Hybrid architecture
    # Every N-th layer is Full Attention.
    # Other layers use Gated DeltaNet.
    # ------------------------------------------------------------------
    full_attention_interval: int = 4

    # ------------------------------------------------------------------
    # Other architectural settings
    # ------------------------------------------------------------------
    attention_dropout: float = 0.0
    use_bias: bool = False
    tie_word_embeddings: bool = True

    initializer_range: float = 0.02

    # Maximum context supported by this Stage.
    max_position_embeddings: int = 512

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def rotary_dim(self) -> int:
        """
        Number of dimensions in each attention head that receive RoPE.
        Qwen3.5 uses partial RoPE.
        """
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def attention_q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def attention_kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def layer_types(self) -> list[str]:
        """
        Example for 8 layers:
        [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        """
        return [
            "full_attention"
            if (layer_idx + 1) % self.full_attention_interval == 0
            else "linear_attention"
            for layer_idx in range(self.num_hidden_layers)
        ]

    def __post_init__(self):
        self._validate()

    def _validate(self):
        # GQA requires Q heads to be divisible by KV heads.
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by "
                "num_key_value_heads"
            )

        # RoPE rotates dimensions in pairs.
        if self.rotary_dim % 2 != 0:
            raise ValueError(
                f"rotary_dim must be even, got {self.rotary_dim}"
            )

        if not (0.0 < self.partial_rotary_factor <= 1.0):
            raise ValueError(
                "partial_rotary_factor must be in (0, 1]"
            )

        if self.num_hidden_layers % self.full_attention_interval != 0:
            raise ValueError(
                "For the current 3:1 hybrid layout, "
                "num_hidden_layers should be divisible by "
                "full_attention_interval."
            )

        if self.linear_conv_kernel_dim <= 0:
            raise ValueError(
                "linear_conv_kernel_dim must be positive"
            )