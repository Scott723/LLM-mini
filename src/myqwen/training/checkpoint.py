from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


def to_plain_dict(value: Any) -> dict[str, Any]:
    """Convert a dataclass, mapping, or Namespace-like object to a plain dict."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Cannot convert {type(value).__name__} to a plain dict")


def save_checkpoint(
    output_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    model_config,
    training_args,
    trainer_state: Mapping[str, Any],
) -> Path:
    """Save model, optimizer, scheduler, config, arguments, and Trainer state."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state = dict(trainer_state)
    if "global_step" not in state:
        raise KeyError("trainer_state must contain 'global_step'")

    global_step = int(state["global_step"])
    if global_step < 0:
        raise ValueError("trainer_state['global_step'] must be non-negative")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": to_plain_dict(model_config),
        "training_args": to_plain_dict(training_args),
        "trainer_state": state,
    }

    path = output_dir / f"step_{global_step:06d}.pt"
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a training checkpoint onto CPU and validate its required fields."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    required = {"model_state_dict", "optimizer_state_dict", "model_config", "training_args"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"Checkpoint is missing required fields: {missing}")

    if "trainer_state" not in checkpoint:
        # Compatibility with checkpoints produced before checkpoint.py existed.
        legacy_state = checkpoint.get("trainer_summary")
        if legacy_state is None:
            raise KeyError("Checkpoint is missing 'trainer_state'")
        checkpoint["trainer_state"] = legacy_state

    return checkpoint


def restore_checkpoint(
    checkpoint: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
) -> dict[str, Any]:
    """Restore model/optimizer/scheduler states and return the saved Trainer state."""
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    saved_scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is None and saved_scheduler_state is not None:
        raise ValueError("Checkpoint contains scheduler state, but scheduler is None")
    if scheduler is not None and saved_scheduler_state is None:
        raise ValueError("A scheduler was provided, but checkpoint has no scheduler state")
    if scheduler is not None:
        scheduler.load_state_dict(saved_scheduler_state)

    trainer_state = dict(checkpoint["trainer_state"])
    for key in ("global_step", "micro_step", "tokens_seen"):
        if key not in trainer_state:
            raise KeyError(f"trainer_state is missing {key!r}")

    return trainer_state
