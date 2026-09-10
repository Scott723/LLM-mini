from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from myqwen.config import ModelConfig
from myqwen.data import (
    TokenShardDataset,
    build_dataloader,
    find_token_shards,
)
from myqwen.modeling import CausalLM
from myqwen.training.evaluator import (
    EvaluatorConfig,
    evaluate_causal_lm,
)


# File location:
#   scripts/eval/eval_stage_a.py
#
# parents[0] -> scripts/eval
# parents[1] -> scripts
# parents[2] -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "fineweb_tokenized_v1"
    / "val"
)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[
    CausalLM,
    ModelConfig,
    dict,
]:
    checkpoint_path = Path(
        checkpoint_path
    ).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint is missing 'model_state_dict'"
        )

    if "model_config" not in checkpoint:
        raise KeyError(
            "Checkpoint is missing 'model_config'"
        )

    model_config = ModelConfig(
        **checkpoint["model_config"]
    )

    model = CausalLM(
        model_config
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    return (
        model,
        model_config,
        checkpoint,
    )


def select_shards(
    data_dir: str | Path,
    pattern: str,
    max_shards: int | None,
) -> list[Path]:
    paths = find_token_shards(
        directory=data_dir,
        pattern=pattern,
    )

    if max_shards is not None:
        if max_shards <= 0:
            raise ValueError(
                "max_shards must be positive or omitted"
            )

        paths = paths[
            :max_shards
        ]

    if not paths:
        raise RuntimeError(
            "No evaluation shards selected"
        )

    return paths


def resolve_seq_len(
    cli_seq_len: int | None,
    model_config: ModelConfig,
    checkpoint: dict,
) -> int:
    if cli_seq_len is not None:
        seq_len = cli_seq_len

    else:
        training_args = checkpoint.get(
            "training_args",
            {},
        )

        checkpoint_seq_len = (
            training_args.get(
                "seq_len"
            )
            if isinstance(
                training_args,
                dict,
            )
            else None
        )

        seq_len = (
            checkpoint_seq_len
            if checkpoint_seq_len is not None
            else model_config.max_position_embeddings
        )

    if seq_len < 2:
        raise ValueError(
            "seq_len must be at least 2"
        )

    if (
        seq_len
        > model_config.max_position_embeddings
    ):
        raise ValueError(
            f"Evaluation seq_len={seq_len} exceeds "
            "model max_position_embeddings="
            f"{model_config.max_position_embeddings}"
        )

    return seq_len


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Stage-A causal language model "
            "checkpoint on validation token shards."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to a Stage-A .pt checkpoint."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(
            DEFAULT_DATA_DIR
        ),
        help=(
            "Directory containing validation .npy shards. "
            "Default: data/fineweb_tokenized_v1/val"
        ),
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="val_*.npy",
        help=(
            "Glob pattern used to select validation shards."
        ),
    )

    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help=(
            "Optionally evaluate only the first N matching shards."
        ),
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help=(
            "Optionally stop after N DataLoader batches."
        ),
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help=(
            "Evaluation sequence length. "
            "Defaults to checkpoint training seq_len "
            "when available, otherwise model max length."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--precision",
        choices=[
            "fp32",
            "bf16",
        ],
        default="bf16",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Optional JSON file for evaluation metrics."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, model_config, checkpoint = (
        load_model_from_checkpoint(
            args.checkpoint
        )
    )

    seq_len = resolve_seq_len(
        cli_seq_len=args.seq_len,
        model_config=model_config,
        checkpoint=checkpoint,
    )

    shard_paths = select_shards(
        data_dir=args.data_dir,
        pattern=args.pattern,
        max_shards=args.max_shards,
    )

    dataset = TokenShardDataset(
        shard_paths=shard_paths,
        seq_len=seq_len,
    )

    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            args.device == "cuda"
            or (
                args.device == "auto"
                and torch.cuda.is_available()
            )
        ),
        drop_last=False,
        seed=None,
    )

    evaluator_config = EvaluatorConfig(
        device=args.device,
        mixed_precision=args.precision,
        max_batches=args.max_batches,
    )

    print(
        "Evaluation plan"
    )

    print(
        f"checkpoint: "
        f"{Path(args.checkpoint).resolve()}"
    )

    print(
        f"data_dir: "
        f"{Path(args.data_dir).resolve()}"
    )

    print(
        f"pattern: {args.pattern}"
    )

    print(
        f"shards: {len(shard_paths)}"
    )

    print(
        f"sequences: {len(dataset):,}"
    )

    print(
        f"seq_len: {seq_len}"
    )

    print(
        f"batch_size: {args.batch_size}"
    )

    print(
        f"precision: {args.precision}"
    )

    print(
        f"device: {args.device}"
    )

    metrics = evaluate_causal_lm(
        model=model,
        dataloader=dataloader,
        config=evaluator_config,
    )

    print(
        "\nEvaluation result"
    )

    print(
        f"loss: "
        f"{metrics['loss']:.6f}"
    )

    print(
        f"perplexity: "
        f"{metrics['perplexity']:.6f}"
    )

    print(
        f"batches: "
        f"{metrics['num_batches']:,}"
    )

    print(
        f"sequences: "
        f"{metrics['num_sequences']:,}"
    )

    print(
        f"prediction_tokens: "
        f"{metrics['prediction_tokens']:,}"
    )

    print(
        f"tokens/s: "
        f"{metrics['tokens_per_second']:,.0f}"
    )

    if args.output_file is not None:
        output_path = Path(
            args.output_file
        ).expanduser().resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metrics,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"saved metrics: "
            f"{output_path}"
        )


if __name__ == "__main__":
    main()
