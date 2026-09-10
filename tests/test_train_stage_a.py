from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "train"
    / "train_stage_a.py"
)


def make_tiny_token_shard(
    directory: Path,
    prefix: str = "train",
) -> Path:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        directory
        / f"{prefix}_00000.npy"
    )

    np.save(
        path,
        np.arange(
            0,
            64,
            dtype=np.uint16,
        ),
    )

    return path


def test_train_stage_a_dry_run(
    tmp_path: Path,
):
    """
    Verify the Stage-A entrypoint can build both train and validation
    DataLoaders without starting an actual model training job.
    """

    train_dir = (
        tmp_path
        / "train"
    )

    val_dir = (
        tmp_path
        / "val"
    )

    make_tiny_token_shard(
        train_dir,
        prefix="train",
    )

    make_tiny_token_shard(
        val_dir,
        prefix="val",
    )

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data-dir",
        str(train_dir),
        "--pattern",
        "train_*.npy",
        "--max-shards",
        "1",
        "--val-data-dir",
        str(val_dir),
        "--val-pattern",
        "val_*.npy",
        "--eval-interval",
        "1",
        "--eval-max-batches",
        "1",
        "--seq-len",
        "8",
        "--batch-size",
        "2",
        "--max-steps",
        "2",
        "--warmup-steps",
        "1",
        "--precision",
        "fp32",
        "--device",
        "cpu",
        "--num-workers",
        "0",
        "--dry-run",
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        result.stdout
        + "\n"
        + result.stderr
    )

    assert (
        "Stage-A training plan"
        in result.stdout
    )

    assert (
        "selected shards: 1"
        in result.stdout
    )

    assert (
        "validation shards: 1"
        in result.stdout
    )

    assert (
        "eval interval: 1 steps"
        in result.stdout
    )

    assert (
        "sample batch shape: (2, 8)"
        in result.stdout
    )

    assert (
        "validation sample batch shape: (2, 8)"
        in result.stdout
    )

    assert (
        "sample batch dtype: torch.int64"
        in result.stdout
    )

    assert (
        "Dry run complete."
        in result.stdout
    )


def test_train_stage_a_rejects_too_long_sequence(
    tmp_path: Path,
):
    """
    Stage-A must reject a sequence length beyond the model limit.
    """

    make_tiny_token_shard(
        tmp_path,
        prefix="train",
    )

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data-dir",
        str(tmp_path),
        "--seq-len",
        "999999",
        "--batch-size",
        "2",
        "--max-steps",
        "2",
        "--warmup-steps",
        "1",
        "--precision",
        "fp32",
        "--device",
        "cpu",
        "--num-workers",
        "0",
        "--no-eval",
        "--dry-run",
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert (
        result.returncode
        != 0
    )

    combined_output = (
        result.stdout
        + "\n"
        + result.stderr
    )

    assert (
        "exceeds config.max_position_embeddings"
        in combined_output
    )
