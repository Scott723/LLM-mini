from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from myqwen.config import ModelConfig
from myqwen.modeling import CausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "eval"
    / "eval_stage_a.py"
)


def test_eval_stage_a_smoke(
    tmp_path: Path,
):
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=8,
        attention_dropout=0.0,
    )

    model = CausalLM(
        config
    )

    checkpoint_path = (
        tmp_path
        / "step_000001.pt"
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "model_config": (
                asdict(config)
            ),
            "training_args": {
                "seq_len": 8,
            },
        },
        checkpoint_path,
    )

    data_dir = (
        tmp_path
        / "val"
    )

    data_dir.mkdir()

    np.save(
        data_dir
        / "val_00000.npy",
        np.arange(
            0,
            64,
            dtype=np.uint16,
        )
        % config.vocab_size,
    )

    output_file = (
        tmp_path
        / "metrics.json"
    )

    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--checkpoint",
        str(checkpoint_path),
        "--data-dir",
        str(data_dir),
        "--pattern",
        "val_*.npy",
        "--batch-size",
        "2",
        "--precision",
        "fp32",
        "--device",
        "cpu",
        "--num-workers",
        "0",
        "--max-batches",
        "2",
        "--output-file",
        str(output_file),
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

    assert output_file.exists()

    metrics = json.loads(
        output_file.read_text(
            encoding="utf-8",
        )
    )

    assert (
        metrics[
            "num_batches"
        ]
        == 2
    )

    assert (
        metrics[
            "loss"
        ]
        > 0.0
    )

    assert (
        metrics[
            "perplexity"
        ]
        > 1.0
    )
