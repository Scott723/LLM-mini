from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Architecture configuration for a decoder-only language model.

    This config only describes model architecture.
    Training hyperparameters such as learning rate, batch size,
    optimizer and training steps should be defined elsewhere.
    """

    # ==============================================================
    # Basic model dimensions
    # ==============================================================

    vocab_size: int

    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int

    # ==============================================================
    # Attention dimensions
    #
    # MHA:
    #   num_attention_heads == num_key_value_heads
    #
    # GQA:
    #   1 < num_key_value_heads < num_attention_heads
    #
    # MQA:
    #   num_key_value_heads == 1
    # ==============================================================

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    # ==============================================================
    # Attention options
    # ==============================================================

    qk_norm: bool = False
    attention_output_gate: bool = False

    attention_bias: bool = False
    attention_dropout: float = 0.0

    # ==============================================================
    # Position embedding
    #
    # Supported:
    #   "rope"
    #   "sinusoidal"
    #
    # partial_rotary_factor:
    #   1.0  -> Full RoPE
    #   <1.0 -> Partial RoPE
    # ==============================================================

    position_embedding_type: str = "rope"

    rope_theta: float = 10_000.0
    partial_rotary_factor: float = 1.0

    max_position_embeddings: int = 512

    # ==============================================================
    # Normalization
    # ==============================================================

    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6

    # ==============================================================
    # Feed Forward Network
    # ==============================================================

    mlp_type: str = "swiglu"
    hidden_act: str = "silu"

    mlp_bias: bool = False

    # ==============================================================
    # Mixer layout
    #
    # Standard Transformer:
    #   ("attention",)
    #
    # Qwen3.5-like hybrid:
    #   (
    #       "gated_deltanet",
    #       "gated_deltanet",
    #       "gated_deltanet",
    #       "attention",
    #   )
    # ==============================================================

    mixer_pattern: tuple[str, ...] = ("attention",)

    # ==============================================================
    # Gated DeltaNet
    #
    # Only used when mixer_pattern contains "gated_deltanet".
    # ==============================================================

    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None

    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None

    linear_conv_kernel_dim: int | None = None

    # ==============================================================
    # Embedding / initialization
    # ==============================================================

    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    # ==============================================================
    # Derived properties
    # ==============================================================

    @property
    def attention_q_dim(self) -> int:
        return (
            self.num_attention_heads
            * self.head_dim
        )

    @property
    def attention_kv_dim(self) -> int:
        return (
            self.num_key_value_heads
            * self.head_dim
        )

    @property
    def rotary_dim(self) -> int:
        return int(
            self.head_dim
            * self.partial_rotary_factor
        )

    @property
    def num_key_value_groups(self) -> int:
        return (
            self.num_attention_heads
            // self.num_key_value_heads
        )

    @property
    def layer_types(self) -> list[str]:
        """
        Expand mixer_pattern to all decoder layers.

        Example:

        mixer_pattern =
            ("gated_deltanet",
             "gated_deltanet",
             "gated_deltanet",
             "attention")

        num_hidden_layers = 8

        gives:

        [
            "gated_deltanet",
            "gated_deltanet",
            "gated_deltanet",
            "attention",
            "gated_deltanet",
            "gated_deltanet",
            "gated_deltanet",
            "attention",
        ]
        """

        pattern_length = len(
            self.mixer_pattern
        )

        return [
            self.mixer_pattern[
                layer_idx % pattern_length
            ]
            for layer_idx
            in range(self.num_hidden_layers)
        ]

    @property
    def linear_key_dim(self) -> int | None:
        if (
            self.linear_num_key_heads is None
            or self.linear_key_head_dim is None
        ):
            return None

        return (
            self.linear_num_key_heads
            * self.linear_key_head_dim
        )

    @property
    def linear_value_dim(self) -> int | None:
        if (
            self.linear_num_value_heads is None
            or self.linear_value_head_dim is None
        ):
            return None

        return (
            self.linear_num_value_heads
            * self.linear_value_head_dim
        )

    # ==============================================================
    # Validation
    # ==============================================================

    def __post_init__(self):
        self._validate()

    def _validate(self):

        # ------------------------------
        # Attention
        # ------------------------------

        if (
            self.num_attention_heads
            % self.num_key_value_heads
            != 0
        ):
            raise ValueError(
                "num_attention_heads must be "
                "divisible by num_key_value_heads"
            )

        if self.head_dim <= 0:
            raise ValueError(
                "head_dim must be positive"
            )

        # ------------------------------
        # Position embedding
        # ------------------------------

        if self.position_embedding_type not in {
            "rope",
            "sinusoidal",
        }:
            raise ValueError(
                "position_embedding_type must be "
                "'rope' or 'sinusoidal'"
            )

        if self.position_embedding_type == "rope":

            if not (
                0.0
                < self.partial_rotary_factor
                <= 1.0
            ):
                raise ValueError(
                    "partial_rotary_factor must "
                    "be in (0, 1]"
                )

            if self.rotary_dim % 2 != 0:
                raise ValueError(
                    "rotary_dim must be even, "
                    f"got {self.rotary_dim}"
                )

        # ------------------------------
        # Norm
        # ------------------------------

        if self.norm_type not in {
            "rmsnorm",
            "layernorm",
        }:
            raise ValueError(
                "Unsupported norm_type: "
                f"{self.norm_type}"
            )

        # ------------------------------
        # MLP
        # ------------------------------

        if self.mlp_type not in {
            "standard",
            "swiglu",
        }:
            raise ValueError(
                "Unsupported mlp_type: "
                f"{self.mlp_type}"
            )

        # ------------------------------
        # Mixer
        # ------------------------------

        if len(self.mixer_pattern) == 0:
            raise ValueError(
                "mixer_pattern cannot be empty"
            )

        # ------------------------------
        # Gated DeltaNet
        # ------------------------------

        if (
            "gated_deltanet"
            in self.mixer_pattern
        ):
            required = {
                "linear_num_key_heads":
                    self.linear_num_key_heads,
                "linear_num_value_heads":
                    self.linear_num_value_heads,
                "linear_key_head_dim":
                    self.linear_key_head_dim,
                "linear_value_head_dim":
                    self.linear_value_head_dim,
                "linear_conv_kernel_dim":
                    self.linear_conv_kernel_dim,
            }

            missing = [
                name
                for name, value
                in required.items()
                if value is None
            ]

            if missing:
                raise ValueError(
                    "Gated DeltaNet requires: "
                    + ", ".join(missing)
                )