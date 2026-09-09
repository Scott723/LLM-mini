from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train" / "train_stage_a.py"


def make_tiny_token_shard(
    directory: Path,
) -> Path:
    path = directory / "train_00000.npy"

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
    Verify that the Stage-A entrypoint can:

        load config_a.py
        discover real .npy-format token data
        build Dataset/DataLoader
        validate the training plan

    without starting an actual model training job.
    """

    make_tiny_token_shard(
        tmp_path
    )

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data-dir",
        str(tmp_path),
        "--pattern",
        "train_*.npy",
        "--max-shards",
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

    assert "Stage-A training plan" in result.stdout
    assert "selected shards: 1" in result.stdout
    assert "sample batch shape: (2, 8)" in result.stdout
    assert "sample batch dtype: torch.int64" in result.stdout
    assert "Dry run complete." in result.stdout


def test_train_stage_a_rejects_too_long_sequence(
    tmp_path: Path,
):
    """
    Stage-A config currently has a finite
    max_position_embeddings. The entrypoint should reject a
    sequence length larger than that configured limit.
    """

    make_tiny_token_shard(
        tmp_path
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
        "--dry-run",
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0

    combined_output = (
        result.stdout
        + "\n"
        + result.stderr
    )

    assert (
        "exceeds config.max_position_embeddings"
        in combined_output
    )
