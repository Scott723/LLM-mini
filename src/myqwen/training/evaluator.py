from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .loss import compute_causal_lm_loss
from .trainer import resolve_device


@dataclass
class EvaluatorConfig:
    device: str = "auto"
    mixed_precision: str = "bf16"
    max_batches: int | None = None

    def __post_init__(self) -> None:
        if self.mixed_precision not in {"fp32", "bf16"}:
            raise ValueError(
                "mixed_precision must be 'fp32' or 'bf16', "
                f"got {self.mixed_precision!r}"
            )
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError("max_batches must be positive or None")


def _autocast_context(device: torch.device, mixed_precision: str):
    if mixed_precision == "bf16":
        if device.type != "cuda":
            raise ValueError(
                "BF16 evaluation is enabled only on CUDA. "
                "Use mixed_precision='fp32' on CPU."
            )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "BF16 evaluation was requested, but this CUDA "
                "device does not report BF16 support."
            )
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    return torch.autocast(device_type=device.type, enabled=False)


def evaluate_causal_lm(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: EvaluatorConfig,
) -> dict[str, float | int]:
    """
    Evaluate a causal LM with token-weighted cross entropy.

    This uses labels=input_ids; compute_causal_lm_loss performs the
    next-token shift internally.
    """
    if len(dataloader) == 0:
        raise ValueError(
            "DataLoader contains zero batches. "
            "Check dataset size and batch_size."
        )

    device = resolve_device(config.device)
    model.to(device)

    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_prediction_tokens = 0
    total_input_tokens = 0
    total_sequences = 0
    num_batches = 0

    start_time = time.perf_counter()

    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(dataloader):
                if (
                    config.max_batches is not None
                    and batch_index >= config.max_batches
                ):
                    break

                if "input_ids" not in batch:
                    raise KeyError(
                        "Evaluation batch must contain an 'input_ids' tensor"
                    )

                input_ids = batch["input_ids"].to(
                    device,
                    non_blocking=True,
                )

                if input_ids.ndim != 2:
                    raise ValueError(
                        "input_ids must have shape [B, T], "
                        f"got {tuple(input_ids.shape)}"
                    )

                batch_size, seq_len = input_ids.shape

                if seq_len < 2:
                    raise ValueError(
                        "Evaluation sequence length must be at least 2"
                    )

                with _autocast_context(
                    device=device,
                    mixed_precision=config.mixed_precision,
                ):
                    logits = model(input_ids)
                    loss = compute_causal_lm_loss(
                        logits=logits,
                        labels=input_ids,
                    )

                if not torch.isfinite(loss.detach()):
                    raise FloatingPointError(
                        f"Non-finite evaluation loss at batch {batch_index}"
                    )

                prediction_tokens = batch_size * (seq_len - 1)

                total_nll += float(loss.detach().item()) * prediction_tokens
                total_prediction_tokens += prediction_tokens
                total_input_tokens += int(input_ids.numel())
                total_sequences += batch_size
                num_batches += 1

    finally:
        if was_training:
            model.train()

    if num_batches == 0:
        raise ValueError("No evaluation batches were processed")

    elapsed_seconds = max(time.perf_counter() - start_time, 1e-12)
    mean_loss = total_nll / total_prediction_tokens

    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = float("inf")

    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "num_batches": num_batches,
        "num_sequences": total_sequences,
        "input_tokens": total_input_tokens,
        "prediction_tokens": total_prediction_tokens,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": total_input_tokens / elapsed_seconds,
    }
