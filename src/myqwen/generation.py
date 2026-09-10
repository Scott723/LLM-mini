from __future__ import annotations

from dataclasses import dataclass

import torch

from myqwen.modeling import CausalLM


@dataclass
class GenerationConfig:
    """Configuration for autoregressive text generation."""

    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    eos_token_id: int | None = None
    use_cache: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or None")


def _sample_next_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not config.do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / config.temperature

    if config.top_k is not None:
        top_k = min(config.top_k, logits.shape[-1])
        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    probabilities = torch.softmax(logits.float(), dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def generate(
    model: CausalLM,
    input_ids: torch.Tensor,
    config: GenerationConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Autoregressively generate tokens with optional KV-cache reuse."""
    config = config or GenerationConfig()

    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids cannot be empty")

    max_total_length = input_ids.shape[1] + config.max_new_tokens
    if max_total_length > model.config.max_position_embeddings:
        raise ValueError(
            f"Requested total length {max_total_length} exceeds "
            f"max_position_embeddings={model.config.max_position_embeddings}"
        )

    if config.eos_token_id is not None:
        if not 0 <= config.eos_token_id < model.config.vocab_size:
            raise ValueError("eos_token_id is outside the model vocabulary")

    model_device = next(model.parameters()).device
    generated = input_ids.to(model_device)
    was_training = model.training
    model.eval()

    finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=model_device)

    try:
        with torch.inference_mode():
            if config.use_cache:
                logits, past_key_values = model(generated, use_cache=True)
            else:
                logits = model(generated)
                past_key_values = None

            for step in range(config.max_new_tokens):
                next_token = _sample_next_token(logits[:, -1, :], config, generator)

                if config.eos_token_id is not None:
                    eos_tokens = torch.full_like(next_token, config.eos_token_id)
                    next_token = torch.where(finished.unsqueeze(-1), eos_tokens, next_token)

                generated = torch.cat((generated, next_token), dim=1)

                if config.eos_token_id is not None:
                    finished |= next_token.squeeze(-1).eq(config.eos_token_id)
                    if bool(finished.all()):
                        break

                if step == config.max_new_tokens - 1:
                    break

                if config.use_cache:
                    logits, past_key_values = model(
                        next_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                else:
                    logits = model(generated)

    finally:
        model.train(was_training)

    return generated
