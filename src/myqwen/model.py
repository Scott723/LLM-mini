import torch
import torch.nn as nn

from myqwen.block import DecoderBlock, build_norm
from myqwen.config import ModelConfig


class DecoderModel(nn.Module):
    """
    Decoder-only language-model backbone.

    Data flow:

        input_ids
            |
            v
        token embedding
            |
            v
        DecoderBlock x N
            |
            v
        final norm
            |
            v
        hidden_states

    Shape:

        input_ids:
            [B, T]

        hidden_states:
            [B, T, hidden_size]

    Notes:
        - RoPE is applied inside the attention module, so no position
          embedding is added to token embeddings in this class.
        - The current end-to-end model path supports RoPE.
        - A fixed sinusoidal input-position embedding can be connected
          here later when that module is implemented.
    """

    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.config = config

        # ----------------------------------------------------------
        # Token embedding
        #
        # [B, T]
        # ->
        # [B, T, hidden_size]
        # ----------------------------------------------------------
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        # ----------------------------------------------------------
        # Decoder stack
        #
        # Every block decides its own mixer type through:
        #
        #     config.layer_types[layer_idx]
        #
        # so the same model class can later support:
        #
        #     attention
        #     linear_attention
        #     gated_deltanet
        #     ...
        # ----------------------------------------------------------
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    config,
                    layer_idx=layer_idx,
                )
                for layer_idx
                in range(config.num_hidden_layers)
            ]
        )

        # ----------------------------------------------------------
        # Final normalization
        #
        # Modern Pre-Norm decoder models generally apply one final
        # normalization after the complete decoder stack.
        # ----------------------------------------------------------
        self.final_norm = build_norm(
            config
        )

        # ----------------------------------------------------------
        # At the moment Stage A uses RoPE.
        #
        # RoPE is handled inside SoftmaxAttention, so DecoderModel
        # only needs to provide position_ids.
        #
        # We explicitly reject unsupported position modes instead of
        # silently applying an incorrect implementation.
        # ----------------------------------------------------------
        if config.position_embedding_type != "rope":
            raise NotImplementedError(
                "DecoderModel currently supports "
                "position_embedding_type='rope' end-to-end. "
                "Connect the corresponding input position "
                "embedding module before using "
                f"'{config.position_embedding_type}'."
            )

        self.apply(
            self._init_weights
        )

    def _init_weights(
        self,
        module: nn.Module,
    ) -> None:
        """
        Initialize trainable projection / embedding weights.

        Linear / Embedding:
            N(0, initializer_range^2)

        Bias:
            0

        Norm parameters keep their own initialization:
            gamma = 1
            beta  = 0 (LayerNorm only)
        """

        if isinstance(
            module,
            nn.Linear,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:
                Token ids with shape [B, T].

            position_ids:
                Optional token positions with shape [B, T] or [T].

                If None, positions are generated as:
                    [0, 1, ..., T - 1]

                Keeping this argument explicit is useful later for
                KV-cache decoding, where the new token may start from
                a non-zero position.

            attention_mask:
                Optional padding mask with shape [B, T].

                True / 1:
                    valid token

                False / 0:
                    padding token

        Returns:
            hidden_states:
                [B, T, hidden_size]
        """

        if input_ids.dim() != 2:
            raise ValueError(
                "input_ids must have shape [B, T], "
                f"got {tuple(input_ids.shape)}"
            )

        batch_size, seq_len = (
            input_ids.shape
        )

        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                "max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )

        # ----------------------------------------------------------
        # Build position ids once and reuse them across all layers.
        #
        # [T]
        # ->
        # [1, T]
        #
        # RoPE broadcasting lets the same position ids be shared by
        # every sample in the batch.
        # ----------------------------------------------------------
        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                device=input_ids.device,
            ).unsqueeze(0)

        # ----------------------------------------------------------
        # Token embedding
        #
        # [B, T]
        # ->
        # [B, T, hidden_size]
        # ----------------------------------------------------------
        hidden_states = (
            self.token_embedding(
                input_ids
            )
        )

        # ----------------------------------------------------------
        # Decoder stack
        # ----------------------------------------------------------
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        # ----------------------------------------------------------
        # Final normalization
        # ----------------------------------------------------------
        hidden_states = self.final_norm(
            hidden_states
        )

        return hidden_states


class CausalLM(nn.Module):
    """
    Decoder-only causal language model.

    Data flow:

        input_ids
            |
            v
        DecoderModel
            |
            v
        hidden_states
            |
            v
        LM Head
            |
            v
        logits

    Shape:

        [B, T]
            ->
        [B, T, hidden_size]
            ->
        [B, T, vocab_size]

    Loss is intentionally not implemented here yet.
    The next step will add next-token cross-entropy training logic.
    """

    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.config = config

        self.model = DecoderModel(
            config
        )

        # ----------------------------------------------------------
        # Language-model head
        #
        # hidden_size -> vocab_size
        #
        # No bias is commonly used for modern decoder-only LMs.
        # ----------------------------------------------------------
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Initialize lm_head before optional weight tying.
        nn.init.normal_(
            self.lm_head.weight,
            mean=0.0,
            std=config.initializer_range,
        )

        # ----------------------------------------------------------
        # Weight tying
        #
        # Input token embedding:
        #     [vocab_size, hidden_size]
        #
        # LM head:
        #     [vocab_size, hidden_size]
        #
        # Both modules therefore can share exactly the same Parameter.
        # ----------------------------------------------------------
        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(
        self,
    ) -> None:
        """
        Share the token-embedding weight with the LM-head weight.
        """

        self.lm_head.weight = (
            self.model.token_embedding.weight
        )

    def get_input_embeddings(
        self,
    ) -> nn.Embedding:
        return (
            self.model.token_embedding
        )

    def get_output_embeddings(
        self,
    ) -> nn.Linear:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            logits:
                [B, T, vocab_size]

        logits are raw, unnormalized scores.
        Do NOT apply softmax here before cross-entropy loss.
        """

        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        logits = self.lm_head(
            hidden_states
        )

        return logits