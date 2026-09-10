from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train" / "pretrain.py"
EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "experiment" / "baseline_19m.py"


def make_tiny_token_shard(directory: Path, prefix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}_00000.npy"
    np.save(path, np.arange(64, dtype=np.uint16))
    return path


def test_pretrain_dry_run(tmp_path: Path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    make_tiny_token_shard(train_dir, "train")
    make_tiny_token_shard(val_dir, "val")

    command = [
        sys.executable,
        str(PRETRAIN_SCRIPT),
        "--experiment",
        str(EXPERIMENT_CONFIG),
        "--data-dir",
        str(train_dir),
        "--val-data-dir",
        str(val_dir),
        "--max-shards",
        "1",
        "--eval-interval",
        "1",
        "--eval-max-batches",
        "1",
        "--seq-len",
        "8",
        "--batch-size",
        "2",
        "--gradient-accumulation-steps",
        "1",
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
        "--no-wandb",
        "--no-save",
        "--dry-run",
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "Pretraining plan" in result.stdout
    assert "experiment: baseline_19m" in result.stdout
    assert "training shards: 1" in result.stdout
    assert "validation shards: 1" in result.stdout
    assert "sample batch shape: (2, 8)" in result.stdout
    assert "validation sample batch shape: (2, 8)" in result.stdout
    assert "Dry run complete." in result.stdout


def test_pretrain_rejects_too_long_sequence(tmp_path: Path):
    make_tiny_token_shard(tmp_path, "train")

    command = [
        sys.executable,
        str(PRETRAIN_SCRIPT),
        "--experiment",
        str(EXPERIMENT_CONFIG),
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
        "--no-wandb",
        "--no-save",
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
    assert "exceeds config.max_position_embeddings" in result.stdout + "\n" + result.stderr
