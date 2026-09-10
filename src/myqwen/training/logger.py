from __future__ import annotations

import importlib
from typing import Any, Mapping, Protocol


class MetricLogger(Protocol):
    """
    Minimal logging interface used by Trainer.

    Trainer only knows about this interface; it does not depend on
    Weights & Biases directly.
    """

    def log(
        self,
        metrics: Mapping[str, Any],
        step: int,
    ) -> None:
        ...

    def finish(self) -> None:
        ...


class NullLogger:
    """
    No-op logger used when experiment tracking is disabled.
    """

    def log(
        self,
        metrics: Mapping[str, Any],
        step: int,
    ) -> None:
        del metrics, step

    def finish(self) -> None:
        return None


class WandbLogger:
    """
    Thin wrapper around Weights & Biases.

    wandb is imported lazily so the package remains optional. Normal
    training and unit tests therefore do not require wandb to be
    installed unless WandB logging is explicitly enabled.
    """

    def __init__(
        self,
        project: str,
        run_name: str | None = None,
        entity: str | None = None,
        mode: str = "online",
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if not project:
            raise ValueError(
                "WandB project name must be non-empty"
            )

        if mode not in {
            "online",
            "offline",
        }:
            raise ValueError(
                "WandB mode must be 'online' or 'offline'"
            )

        try:
            wandb = importlib.import_module(
                "wandb"
            )
        except ImportError as error:
            raise ImportError(
                "WandB logging was enabled, but the 'wandb' "
                "package is not installed. Install it with:\n"
                "    python -m pip install wandb"
            ) from error

        self._wandb = wandb

        init_kwargs: dict[str, Any] = {
            "project": project,
            "mode": mode,
        }

        if run_name is not None:
            init_kwargs[
                "name"
            ] = run_name

        if entity is not None:
            init_kwargs[
                "entity"
            ] = entity

        if config is not None:
            init_kwargs[
                "config"
            ] = dict(
                config
            )

        self._run = wandb.init(
            **init_kwargs
        )

        if self._run is None:
            raise RuntimeError(
                "wandb.init() returned None"
            )

    def log(
        self,
        metrics: Mapping[str, Any],
        step: int,
    ) -> None:
        if step < 0:
            raise ValueError(
                "step must be non-negative"
            )

        self._run.log(
            dict(
                metrics
            ),
            step=step,
        )

    def finish(self) -> None:
        self._run.finish()


def build_logger(
    *,
    enable_wandb: bool,
    project: str = "llm-mini",
    run_name: str | None = None,
    entity: str | None = None,
    mode: str = "online",
    config: Mapping[str, Any] | None = None,
) -> MetricLogger:
    """
    Build the experiment logger.

    When enable_wandb=False, returns NullLogger and does not import
    wandb at all.
    """

    if not enable_wandb:
        return NullLogger()

    return WandbLogger(
        project=project,
        run_name=run_name,
        entity=entity,
        mode=mode,
        config=config,
    )
