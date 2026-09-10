from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from myqwen.training.logger import MetricLogger, NullLogger
from myqwen.training.loss import compute_causal_lm_loss
from myqwen.training.profiling import (
    achieved_tflops,
    estimate_mfu,
    estimate_training_flops_per_token,
    get_cuda_peak_memory,
    reset_cuda_peak_memory,
)


CheckpointCallback = Callable[[dict[str, object]], None]


@dataclass
class TrainerConfig:
    """Configuration for the single-GPU causal-LM trainer."""

    max_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 1.0
    mixed_precision: str = "bf16"
    device: str = "auto"
    log_interval: int = 10
    eval_interval: int | None = None
    eval_max_batches: int | None = 500
    save_interval: int | None = None
    device_peak_tflops: float | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        if self.mixed_precision not in {"fp32", "bf16"}:
            raise ValueError("mixed_precision must be 'fp32' or 'bf16'")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.eval_interval is not None and self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive or None")
        if self.eval_max_batches is not None and self.eval_max_batches <= 0:
            raise ValueError("eval_max_batches must be positive or None")
        if self.save_interval is not None and self.save_interval <= 0:
            raise ValueError("save_interval must be positive or None")
        if self.device_peak_tflops is not None and self.device_peak_tflops <= 0:
            raise ValueError("device_peak_tflops must be positive or None")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return resolved


def cycle_dataloader(dataloader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    if len(dataloader) == 0:
        raise ValueError("DataLoader contains zero batches")
    while True:
        yield from dataloader


class Trainer:
    """Single-process causal-LM trainer with validation, logging, resume, and profiling."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        dataloader: DataLoader,
        config: TrainerConfig,
        scheduler: LRScheduler | None = None,
        eval_dataloader: DataLoader | None = None,
        logger: MetricLogger | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.config = config
        self.scheduler = scheduler
        self.eval_dataloader = eval_dataloader
        self.logger = logger if logger is not None else NullLogger()
        self.checkpoint_callback = checkpoint_callback

        self.device = resolve_device(config.device)
        if config.mixed_precision == "bf16" and self.device.type != "cuda":
            raise ValueError("BF16 training is enabled only on CUDA")
        if config.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 training was requested, but this CUDA device does not support BF16")
        if config.eval_interval is not None and eval_dataloader is None:
            raise ValueError("eval_interval is enabled, but eval_dataloader is None")
        if config.save_interval is not None and checkpoint_callback is None:
            raise ValueError("save_interval is enabled, but checkpoint_callback is None")

        self.model.to(self.device)

        self.global_step = 0
        self.micro_step = 0
        self.tokens_seen = 0
        self.validation_history: list[dict[str, float | int]] = []

        self._flops_per_token: float | None = None
        self._profile_seq_len: int | None = None
        self._run_peak_allocated_gib = 0.0
        self._run_peak_reserved_gib = 0.0

    def state_dict(self) -> dict[str, object]:
        """Return the minimal Trainer state required to continue training."""
        return {
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "tokens_seen": self.tokens_seen,
            "validation_history": list(self.validation_history),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore Trainer counters/history before calling train()."""
        for key in ("global_step", "micro_step", "tokens_seen"):
            if key not in state:
                raise KeyError(f"Trainer state is missing {key!r}")

        global_step = int(state["global_step"])
        micro_step = int(state["micro_step"])
        tokens_seen = int(state["tokens_seen"])

        if global_step < 0 or micro_step < 0 or tokens_seen < 0:
            raise ValueError("Trainer counters must be non-negative")
        if global_step > self.config.max_steps:
            raise ValueError(
                f"Checkpoint global_step={global_step} exceeds max_steps={self.config.max_steps}"
            )

        self.global_step = global_step
        self.micro_step = micro_step
        self.tokens_seen = tokens_seen
        self.validation_history = list(state.get("validation_history", []))

    def _autocast_context(self):
        if self.config.mixed_precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type=self.device.type, enabled=False)

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "input_ids" not in batch:
            raise KeyError("Training batch must contain an 'input_ids' tensor")
        input_ids = batch["input_ids"]
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        return input_ids.to(self.device, non_blocking=True)

    def _prepare_mfu_estimator(self, seq_len: int) -> None:
        if self.config.device_peak_tflops is None:
            return

        if self._profile_seq_len is not None and seq_len != self._profile_seq_len:
            raise ValueError(
                f"MFU estimator expected seq_len={self._profile_seq_len}, got seq_len={seq_len}"
            )

        if self._flops_per_token is None:
            self._flops_per_token = estimate_training_flops_per_token(self.model, seq_len)
            self._profile_seq_len = seq_len

    def _update_training_peak_memory(self):
        peak_memory = get_cuda_peak_memory(self.device)
        if peak_memory is None:
            return None

        self._run_peak_allocated_gib = max(
            self._run_peak_allocated_gib, peak_memory.allocated_gib
        )
        self._run_peak_reserved_gib = max(
            self._run_peak_reserved_gib, peak_memory.reserved_gib
        )
        return peak_memory

    def _should_evaluate(self) -> bool:
        if self.config.eval_interval is None or self.eval_dataloader is None:
            return False
        return self.global_step % self.config.eval_interval == 0 or self.global_step == self.config.max_steps

    def _should_save_periodic_checkpoint(self) -> bool:
        if self.config.save_interval is None or self.checkpoint_callback is None:
            return False
        return self.global_step % self.config.save_interval == 0 and self.global_step < self.config.max_steps

    def _run_validation(self) -> dict[str, float | int]:
        if self.eval_dataloader is None:
            raise RuntimeError("eval_dataloader is required for validation")

        from myqwen.training.evaluator import EvaluatorConfig, evaluate_causal_lm

        metrics = evaluate_causal_lm(
            model=self.model,
            dataloader=self.eval_dataloader,
            config=EvaluatorConfig(
                device=str(self.device),
                mixed_precision=self.config.mixed_precision,
                max_batches=self.config.eval_max_batches,
            ),
        )

        record: dict[str, float | int] = {"step": self.global_step, **metrics}
        self.validation_history.append(record)

        print(
            f"validation step {self.global_step:>6d} | loss {metrics['loss']:.4f} | "
            f"ppl {metrics['perplexity']:.2f} | batches {metrics['num_batches']:,} | "
            f"tokens/s {metrics['tokens_per_second']:,.0f}"
        )

        self.logger.log(
            {
                "val/loss": metrics["loss"],
                "val/perplexity": metrics["perplexity"],
                "val/tokens_per_second": metrics["tokens_per_second"],
                "val/prediction_tokens": metrics["prediction_tokens"],
                "train/tokens_seen": self.tokens_seen,
            },
            step=self.global_step,
        )
        return record

    def train(self) -> dict[str, object]:
        self.model.train()
        batch_iterator = cycle_dataloader(self.dataloader)
        self.optimizer.zero_grad(set_to_none=True)
        reset_cuda_peak_memory(self.device)

        train_start_time = time.perf_counter()
        start_tokens_seen = self.tokens_seen
        log_start_time = train_start_time
        interval_loss_sum = 0.0
        interval_tokens = 0
        interval_step_count = 0
        total_validation_seconds = 0.0

        print("Training started")
        print(
            f"device={self.device} | precision={self.config.mixed_precision} | "
            f"start_step={self.global_step} | max_steps={self.config.max_steps} | "
            f"grad_accum={self.config.gradient_accumulation_steps}"
        )
        if self.config.device_peak_tflops is None:
            print("MFU: disabled (set device_peak_tflops to enable)")
        else:
            print(f"MFU: enabled | device peak={self.config.device_peak_tflops:.2f} TFLOPs")

        if self.config.eval_interval is not None:
            print(
                f"validation enabled | interval={self.config.eval_interval} steps | "
                f"max_batches={self.config.eval_max_batches}"
            )
        if self.config.save_interval is not None:
            print(f"periodic checkpointing enabled | interval={self.config.save_interval} steps")

        while self.global_step < self.config.max_steps:
            accumulated_loss = 0.0
            optimizer_step_tokens = 0

            for _ in range(self.config.gradient_accumulation_steps):
                batch = next(batch_iterator)
                input_ids = self._move_batch(batch)
                micro_batch_tokens = int(input_ids.numel())
                self._prepare_mfu_estimator(input_ids.shape[1])

                with self._autocast_context():
                    logits = self.model(input_ids)
                    raw_loss = compute_causal_lm_loss(logits=logits, labels=input_ids)
                    loss = raw_loss / self.config.gradient_accumulation_steps

                if not torch.isfinite(raw_loss.detach()):
                    raise FloatingPointError(
                        f"Non-finite loss at global_step={self.global_step}, micro_step={self.micro_step}"
                    )

                loss.backward()
                accumulated_loss += float(raw_loss.detach().item())
                optimizer_step_tokens += micro_batch_tokens
                self.tokens_seen += micro_batch_tokens
                self.micro_step += 1

            step_loss = accumulated_loss / self.config.gradient_accumulation_steps
            grad_norm = None

            if self.config.max_grad_norm is not None:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.max_grad_norm
                )
                grad_norm = float(grad_norm_tensor.detach().item())
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient norm at global_step={self.global_step}")

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1
            interval_loss_sum += step_loss
            interval_tokens += optimizer_step_tokens
            interval_step_count += 1

            should_evaluate = self._should_evaluate()
            should_log = (
                self.global_step % self.config.log_interval == 0
                or self.global_step == 1
                or self.global_step == self.config.max_steps
                or should_evaluate
            )

            if should_log:
                now = time.perf_counter()
                elapsed = max(now - log_start_time, 1e-12)
                tokens_per_second = interval_tokens / elapsed
                mean_interval_loss = interval_loss_sum / interval_step_count
                lr = float(self.optimizer.param_groups[0]["lr"])

                message = (
                    f"step {self.global_step:>6d}/{self.config.max_steps} | "
                    f"loss {mean_interval_loss:.4f} | lr {lr:.3e} | "
                    f"tokens/s {tokens_per_second:,.0f}"
                )
                if grad_norm is not None:
                    message += f" | grad_norm {grad_norm:.4f}"

                train_metrics: dict[str, float | int] = {
                    "train/loss": mean_interval_loss,
                    "train/lr": lr,
                    "train/tokens_per_second": tokens_per_second,
                    "train/tokens_seen": self.tokens_seen,
                }
                if grad_norm is not None:
                    train_metrics["train/grad_norm"] = grad_norm

                peak_memory = self._update_training_peak_memory()
                if peak_memory is not None:
                    train_metrics["system/peak_memory_allocated_gib"] = self._run_peak_allocated_gib
                    train_metrics["system/peak_memory_reserved_gib"] = self._run_peak_reserved_gib
                    message += f" | peak_mem {self._run_peak_allocated_gib:.2f} GiB"

                if self._flops_per_token is not None and self.config.device_peak_tflops is not None:
                    current_tflops = achieved_tflops(tokens_per_second, self._flops_per_token)
                    current_mfu = estimate_mfu(
                        tokens_per_second,
                        self._flops_per_token,
                        self.config.device_peak_tflops,
                    )
                    train_metrics["train/achieved_tflops"] = current_tflops
                    train_metrics["train/mfu"] = current_mfu
                    message += f" | MFU {current_mfu * 100:.1f}%"

                print(message)
                self.logger.log(train_metrics, step=self.global_step)

                interval_loss_sum = 0.0
                interval_tokens = 0
                interval_step_count = 0
                log_start_time = now

            if should_evaluate:
                validation_metrics = self._run_validation()
                total_validation_seconds += float(validation_metrics["elapsed_seconds"])
                reset_cuda_peak_memory(self.device)
                log_start_time = time.perf_counter()

            if self._should_save_periodic_checkpoint():
                self.checkpoint_callback(self.state_dict())

        wall_elapsed = max(time.perf_counter() - train_start_time, 1e-12)
        training_elapsed = max(wall_elapsed - total_validation_seconds, 1e-12)
        run_tokens = self.tokens_seen - start_tokens_seen
        average_tokens_per_second = run_tokens / training_elapsed

        summary: dict[str, object] = {
            **self.state_dict(),
            "elapsed_seconds": wall_elapsed,
            "validation_seconds": total_validation_seconds,
            "average_tokens_per_second": average_tokens_per_second,
        }

        peak_memory = self._update_training_peak_memory()
        if peak_memory is not None:
            summary["peak_memory_allocated_gib"] = self._run_peak_allocated_gib
            summary["peak_memory_reserved_gib"] = self._run_peak_reserved_gib

        if self._flops_per_token is not None and self.config.device_peak_tflops is not None:
            summary["estimated_flops_per_token"] = self._flops_per_token
            summary["average_achieved_tflops"] = achieved_tflops(
                average_tokens_per_second, self._flops_per_token
            )
            summary["average_mfu"] = estimate_mfu(
                average_tokens_per_second,
                self._flops_per_token,
                self.config.device_peak_tflops,
            )

        print("Training finished")
        print(
            f"steps={self.global_step} | tokens={self.tokens_seen:,} | elapsed={wall_elapsed:.2f}s | "
            f"avg_tokens/s={average_tokens_per_second:,.0f}"
        )

        if peak_memory is not None:
            print(
                f"peak training CUDA memory | allocated={self._run_peak_allocated_gib:.2f} GiB | "
                f"reserved={self._run_peak_reserved_gib:.2f} GiB"
            )

        if "average_mfu" in summary:
            print(
                f"average compute | achieved={summary['average_achieved_tflops']:.2f} TFLOPs | "
                f"MFU={summary['average_mfu'] * 100:.1f}%"
            )

        if self.validation_history:
            last_validation = self.validation_history[-1]
            print(
                f"final validation | step={last_validation['step']} | "
                f"loss={last_validation['loss']:.4f} | ppl={last_validation['perplexity']:.2f}"
            )

        return summary
