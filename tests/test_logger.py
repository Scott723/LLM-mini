from __future__ import annotations

from types import SimpleNamespace

import pytest

import myqwen.training.logger as logger_module
from myqwen.training.logger import (
    NullLogger,
    WandbLogger,
    build_logger,
)


class FakeRun:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(
        self,
        metrics,
        step,
    ):
        self.logged.append(
            {
                "metrics": dict(
                    metrics
                ),
                "step": step,
            }
        )

    def finish(self):
        self.finished = True


def test_build_logger_disabled_returns_null_logger():
    logger = build_logger(
        enable_wandb=False
    )

    assert isinstance(
        logger,
        NullLogger,
    )

    # No-op behavior should be safe.
    logger.log(
        {
            "train/loss": 1.0,
        },
        step=1,
    )

    logger.finish()


def test_wandb_logger_initializes_logs_and_finishes(
    monkeypatch,
):
    fake_run = FakeRun()

    init_calls = []

    fake_wandb = SimpleNamespace()

    def fake_init(
        **kwargs,
    ):
        init_calls.append(
            kwargs
        )

        return fake_run

    fake_wandb.init = (
        fake_init
    )

    def fake_import_module(
        name: str,
    ):
        assert (
            name
            == "wandb"
        )

        return fake_wandb

    monkeypatch.setattr(
        logger_module.importlib,
        "import_module",
        fake_import_module,
    )

    logger = WandbLogger(
        project="llm-mini",
        run_name="unit-test",
        entity="test-entity",
        mode="offline",
        config={
            "max_steps": 4,
        },
    )

    assert len(
        init_calls
    ) == 1

    assert (
        init_calls[0][
            "project"
        ]
        == "llm-mini"
    )

    assert (
        init_calls[0][
            "name"
        ]
        == "unit-test"
    )

    assert (
        init_calls[0][
            "entity"
        ]
        == "test-entity"
    )

    assert (
        init_calls[0][
            "mode"
        ]
        == "offline"
    )

    logger.log(
        {
            "train/loss": 2.5,
            "train/lr": 1e-3,
        },
        step=7,
    )

    assert (
        fake_run.logged
        == [
            {
                "metrics": {
                    "train/loss": 2.5,
                    "train/lr": 1e-3,
                },
                "step": 7,
            }
        ]
    )

    logger.finish()

    assert (
        fake_run.finished
        is True
    )


def test_wandb_logger_rejects_invalid_mode():
    with pytest.raises(
        ValueError,
        match="mode",
    ):
        WandbLogger(
            project="llm-mini",
            mode="invalid",
        )
