from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from myqwen.config import ModelConfig
from myqwen.modeling.attention.full_attention import KVCache, VALID_ATTENTION_BACKENDS
from myqwen.modeling.block import DecoderBlock, build_norm


PastKeyValues = tuple[KVCache | None, ...]


class DecoderModel(nn.Module):
    """Decoder-only language-model backbone with optional KV-cache decoding."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_norm = build_norm(config)
        self.activation_checkpointing = False

        if config.position_embedding_type != "rope":
            raise NotImplementedError(
                "DecoderModel currently supports position_embedding_type='rope' end-to-end"
            )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def set_attention_backend(self, backend: str) -> None:
        """Switch the runtime implementation used by full-attention layers."""
        backend = backend.lower()

        if backend not in VALID_ATTENTION_BACKENDS:
            raise ValueError(
                f"attention backend must be one of {sorted(VALID_ATTENTION_BACKENDS)}, got {backend!r}"
            )

        for layer in self.layers:
            setter = getattr(layer.mixer, "set_attention_backend", None)
            if setter is not None:
                setter(backend)

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """Enable layer-wise activation checkpointing during training."""
        self.activation_checkpointing = bool(enabled)

    def _prepare_past_key_values(
        self,
        past_key_values: PastKeyValues | None,
        use_cache: bool,
    ) -> tuple[PastKeyValues, int]:
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")

        if past_key_values is None:
            return tuple(None for _ in self.layers), 0

        if len(past_key_values) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} layer caches, got {len(past_key_values)}")

        past_length = None

        for layer_cache in past_key_values:
            if layer_cache is None:
                continue

            key, value = layer_cache

            if key.shape[-2] != value.shape[-2]:
                raise ValueError("Cached K and V must have the same sequence length")

            layer_past_length = key.shape[-2]

            if past_length is None:
                past_length = layer_past_length
            elif layer_past_length != past_length:
                raise ValueError("All layer KV caches must have the same sequence length")

        return past_key_values, 0 if past_length is None else past_length

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ):
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")

        batch_size, seq_len = input_ids.shape
        layer_caches, past_length = self._prepare_past_key_values(past_key_values, use_cache)
        total_length = past_length + seq_len

        if total_length > self.config.max_position_embeddings:
            raise ValueError(
                f"Total sequence length {total_length} exceeds "
                f"max_position_embeddings={self.config.max_position_embeddings}"
            )

        if attention_mask is not None and attention_mask.shape != (batch_size, total_length):
            raise ValueError(
                f"attention_mask must have shape {(batch_size, total_length)}, "
                f"got {tuple(attention_mask.shape)}"
            )

        if position_ids is None:
            position_ids = torch.arange(past_length, total_length, device=input_ids.device).unsqueeze(0)

        hidden_states = self.token_embedding(input_ids)
        present_key_values = []

        for layer, layer_cache in zip(self.layers, layer_caches):
            if self.activation_checkpointing and self.training and not use_cache:
                block_forward = partial(layer, position_ids=position_ids, attention_mask=attention_mask)
                hidden_states = activation_checkpoint(block_forward, hidden_states, use_reentrant=False)
            elif use_cache:
                hidden_states, present_key_value = layer(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_value=layer_cache,
                    use_cache=True,
                )
                present_key_values.append(present_key_value)
            else:
                hidden_states = layer(hidden_states, position_ids=position_ids, attention_mask=attention_mask)

        hidden_states = self.final_norm(hidden_states)

        if use_cache:
            return hidden_states, tuple(present_key_values)

        return hidden_states


class CausalLM(nn.Module):
    """Decoder-only causal language model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = DecoderModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.initializer_range)

        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.token_embedding.weight

    def set_attention_backend(self, backend: str) -> None:
        self.model.set_attention_backend(backend)

    def set_activation_checkpointing(self, enabled: bool) -> None:
        self.model.set_activation_checkpointing(enabled)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.token_embedding

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ):
        if use_cache:
            hidden_states, present_key_values = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            return self.lm_head(hidden_states), present_key_values

        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        return self.lm_head(hidden_states)
