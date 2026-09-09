from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from myqwen.training.loss import compute_causal_lm_loss


@dataclass
class TrainerConfig:
    """
    Configuration for the first single-GPU causal-LM trainer.

    Notes
    -----
    max_steps counts optimizer updates, not micro-batches.

    Example:
        gradient_accumulation_steps = 4
        max_steps = 100

    means:
        400 forward/backward micro-steps
        100 optimizer updates
    """

    max_steps: int = 100
    gradient_accumulation_steps: int = 1

    # None disables gradient clipping.
    max_grad_norm: float | None = 1.0

    # Supported in v1:
    #   "fp32"
    #   "bf16"
    mixed_precision: str = "bf16"

    # "auto" -> CUDA if available, otherwise CPU.
    device: str = "auto"

    log_interval: int = 10

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(
                f"max_steps must be positive, got {self.max_steps}"
            )

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be positive, "
                f"got {self.gradient_accumulation_steps}"
            )

        if (
            self.max_grad_norm is not None
            and self.max_grad_norm <= 0
        ):
            raise ValueError(
                "max_grad_norm must be positive or None"
            )

        if self.mixed_precision not in {
            "fp32",
            "bf16",
        }:
            raise ValueError(
                "mixed_precision must be 'fp32' or 'bf16', "
                f"got {self.mixed_precision!r}"
            )

        if self.log_interval <= 0:
            raise ValueError(
                f"log_interval must be positive, got {self.log_interval}"
            )


def resolve_device(
    device: str,
) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    resolved = torch.device(device)

    if (
        resolved.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False"
        )

    return resolved


def cycle_dataloader(
    dataloader: DataLoader,
) -> Iterator[dict[str, torch.Tensor]]:
    """
    Yield batches forever.

    When one DataLoader epoch is exhausted, creating a new iterator
    starts the next epoch. If shuffle=True, PyTorch reshuffles the
    sample order for the new epoch.
    """

    if len(dataloader) == 0:
        raise ValueError(
            "DataLoader contains zero batches. "
            "Check dataset size, batch_size, and drop_last."
        )

    while True:
        for batch in dataloader:
            yield batch


class Trainer:
    """
    Minimal single-process trainer for causal language-model pretraining.

    The trainer deliberately does NOT construct the optimizer or scheduler.
    Those are separate concerns handled by myqwen.optim.

    Current training path:

        DataLoader
            ->
        input_ids
            ->
        model
            ->
        logits
            ->
        causal LM loss
            ->
        backward
            ->
        optional gradient clipping
            ->
        optimizer.step()
            ->
        scheduler.step()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        dataloader: DataLoader,
        config: TrainerConfig,
        scheduler: LRScheduler | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.config = config
        self.scheduler = scheduler

        self.device = resolve_device(
            config.device
        )

        if (
            config.mixed_precision == "bf16"
            and self.device.type != "cuda"
        ):
            raise ValueError(
                "The v1 trainer enables bf16 only on CUDA. "
                "Use mixed_precision='fp32' on CPU."
            )

        if (
            config.mixed_precision == "bf16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "BF16 training was requested, but this CUDA device "
                "does not report BF16 support."
            )

        self.model.to(
            self.device
        )

        self.global_step = 0
        self.micro_step = 0
        self.tokens_seen = 0

    def _autocast_context(self):
        """
        Return the appropriate autocast context.

        BF16 uses autocast for matrix-heavy operations while parameters
        can remain in their normal model dtype.
        """

        if self.config.mixed_precision == "bf16":
            return torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )

        # A disabled autocast context keeps FP32 behavior.
        return torch.autocast(
            device_type=self.device.type,
            enabled=False,
        )

    def _move_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "input_ids" not in batch:
            raise KeyError(
                "Training batch must contain an 'input_ids' tensor"
            )

        input_ids = batch[
            "input_ids"
        ]

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, seq_len], "
                f"got {tuple(input_ids.shape)}"
            )

        return input_ids.to(
            self.device,
            non_blocking=True,
        )

    def train(
        self,
    ) -> dict[str, float | int]:
        self.model.train()

        batch_iterator = cycle_dataloader(
            self.dataloader
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        train_start_time = time.perf_counter()
        log_start_time = train_start_time

        interval_loss_sum = 0.0
        interval_tokens = 0

        print(
            "Training started"
        )
        print(
            f"device={self.device} | "
            f"precision={self.config.mixed_precision} | "
            f"max_steps={self.config.max_steps} | "
            f"grad_accum={self.config.gradient_accumulation_steps}"
        )

        while (
            self.global_step
            < self.config.max_steps
        ):
            accumulated_loss = 0.0
            optimizer_step_tokens = 0

            for _ in range(
                self.config.gradient_accumulation_steps
            ):
                batch = next(
                    batch_iterator
                )

                input_ids = self._move_batch(
                    batch
                )

                micro_batch_tokens = int(
                    input_ids.numel()
                )

                with self._autocast_context():
                    logits = self.model(
                        input_ids
                    )

                    raw_loss = (
                        compute_causal_lm_loss(
                            logits=logits,
                            labels=input_ids,
                        )
                    )

                    # Divide before backward so accumulated gradients
                    # equal the mean gradient across micro-batches.
                    loss = (
                        raw_loss
                        / self.config.gradient_accumulation_steps
                    )

                if not torch.isfinite(
                    raw_loss.detach()
                ):
                    raise FloatingPointError(
                        "Non-finite loss encountered at "
                        f"global_step={self.global_step}, "
                        f"micro_step={self.micro_step}: "
                        f"{raw_loss.detach().item()}"
                    )

                loss.backward()

                raw_loss_value = float(
                    raw_loss.detach().item()
                )

                accumulated_loss += (
                    raw_loss_value
                )

                optimizer_step_tokens += (
                    micro_batch_tokens
                )

                self.tokens_seen += (
                    micro_batch_tokens
                )

                self.micro_step += 1

            # Average raw loss across the micro-batches belonging to
            # this optimizer update. This is only for logging.
            step_loss = (
                accumulated_loss
                / self.config.gradient_accumulation_steps
            )

            grad_norm = None

            if (
                self.config.max_grad_norm
                is not None
            ):
                grad_norm_tensor = (
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.max_grad_norm,
                    )
                )

                grad_norm = float(
                    grad_norm_tensor.detach().item()
                )

                if not math.isfinite(
                    grad_norm
                ):
                    raise FloatingPointError(
                        "Non-finite gradient norm encountered at "
                        f"global_step={self.global_step}"
                    )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad(
                set_to_none=True
            )

            self.global_step += 1

            interval_loss_sum += (
                step_loss
            )

            interval_tokens += (
                optimizer_step_tokens
            )

            should_log = (
                self.global_step
                % self.config.log_interval
                == 0
                or self.global_step
                == 1
                or self.global_step
                == self.config.max_steps
            )

            if should_log:
                now = time.perf_counter()

                elapsed = max(
                    now - log_start_time,
                    1e-12,
                )

                tokens_per_second = (
                    interval_tokens
                    / elapsed
                )

                # Number of optimizer steps represented by the
                # current logging interval.
                if self.global_step == 1:
                    interval_steps = 1
                else:
                    interval_steps = min(
                        self.config.log_interval,
                        self.global_step,
                    )

                mean_interval_loss = (
                    interval_loss_sum
                    / interval_steps
                )

                lr = float(
                    self.optimizer.param_groups[0][
                        "lr"
                    ]
                )

                message = (
                    f"step {self.global_step:>6d}/"
                    f"{self.config.max_steps} | "
                    f"loss {mean_interval_loss:.4f} | "
                    f"lr {lr:.3e} | "
                    f"tokens/s {tokens_per_second:,.0f}"
                )

                if grad_norm is not None:
                    message += (
                        f" | grad_norm {grad_norm:.4f}"
                    )

                print(
                    message
                )

                interval_loss_sum = 0.0
                interval_tokens = 0
                log_start_time = now

        total_elapsed = max(
            time.perf_counter()
            - train_start_time,
            1e-12,
        )

        summary = {
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "tokens_seen": self.tokens_seen,
            "elapsed_seconds": total_elapsed,
            "average_tokens_per_second": (
                self.tokens_seen
                / total_elapsed
            ),
        }

        print(
            "Training finished"
        )
        print(
            f"steps={self.global_step} | "
            f"tokens={self.tokens_seen:,} | "
            f"elapsed={total_elapsed:.2f}s | "
            f"avg_tokens/s="
            f"{summary['average_tokens_per_second']:,.0f}"
        )

        return summary
