from __future__ import annotations

import argparse
import runpy
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

from myqwen.data import (
    TokenShardDataset,
    build_dataloader,
    find_token_shards,
)

from myqwen.modeling import CausalLM

from myqwen.training import (
    build_lr_scheduler,
    build_optimizer,
    Trainer,
    TrainerConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config_a.py"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "fineweb_tokenized"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "stage_a"


def load_model_config(
    config_path: str | Path,
):
    """
    Load MODEL_CONFIG from a Python config file.

    runpy is used deliberately so this script can still be launched as:

        python scripts/train_stage_a.py

    without requiring the top-level configs/ directory to be an
    installed Python package.
    """

    config_path = Path(
        config_path
    ).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config file not found: {config_path}"
        )

    namespace = runpy.run_path(
        str(config_path)
    )

    if "MODEL_CONFIG" not in namespace:
        raise KeyError(
            f"{config_path} must define MODEL_CONFIG"
        )

    return namespace["MODEL_CONFIG"]


def select_shards(
    data_dir: str | Path,
    pattern: str,
    max_shards: int | None,
) -> list[Path]:
    """
    Find tokenized shards and optionally keep only the first N.

    max_shards is useful for smoke training without changing the
    underlying dataset directory.
    """

    paths = find_token_shards(
        directory=data_dir,
        pattern=pattern,
    )

    if max_shards is not None:
        if max_shards <= 0:
            raise ValueError(
                "max_shards must be positive or omitted"
            )

        paths = paths[:max_shards]

    if len(paths) == 0:
        raise RuntimeError(
            "No token shards selected"
        )

    return paths


def count_parameters(
    model: torch.nn.Module,
) -> tuple[int, int]:
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def save_checkpoint(
    output_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    model_config,
    training_args: argparse.Namespace,
    trainer_summary: dict,
) -> Path:
    """
    Save the final Stage-A training state.

    Resume logic is intentionally not implemented yet, but optimizer
    and scheduler states are stored so the checkpoint format already
    contains the information needed for it later.
    """

    output_dir = Path(
        output_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if is_dataclass(model_config):
        model_config_state = asdict(
            model_config
        )
    else:
        model_config_state = dict(
            vars(model_config)
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "model_config": model_config_state,
        "training_args": vars(training_args),
        "trainer_summary": trainer_summary,
    }

    path = (
        output_dir
        / f"step_{trainer_summary['global_step']:06d}.pt"
    )

    torch.save(
        checkpoint,
        path,
    )

    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-A pretraining for the configurable decoder-only "
            "language model."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Python file defining MODEL_CONFIG.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing tokenized .npy shards.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="train_*.npy",
        help="Glob pattern for token shards.",
    )

    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help=(
            "Use only the first N token shards. "
            "Omit to use all matching shards."
        ),
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help=(
            "Training sequence length. "
            "Default: config.max_position_embeddings."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Micro-batch size.",
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of micro-batches per optimizer update.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Number of optimizer updates.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help=(
            "Final cosine LR as a fraction of the base LR."
        ),
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="bf16",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Examples: auto, cuda, cuda:0, cpu.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--fused-adamw",
        action="store_true",
        help="Use PyTorch fused AdamW when supported.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the final checkpoint.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate config/data and print the training plan "
            "without constructing the model or starting training."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    if args.max_steps <= 0:
        raise ValueError(
            "max_steps must be positive"
        )

    if args.warmup_steps < 0:
        raise ValueError(
            "warmup_steps must be >= 0"
        )

    if args.warmup_steps > args.max_steps:
        raise ValueError(
            "warmup_steps cannot exceed max_steps"
        )

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    model_config = load_model_config(
        args.config
    )

    seq_len = (
        args.seq_len
        if args.seq_len is not None
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
            f"seq_len={seq_len} exceeds "
            "config.max_position_embeddings="
            f"{model_config.max_position_embeddings}"
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

    if len(dataset) < args.batch_size:
        raise ValueError(
            f"Dataset has only {len(dataset)} sequences, "
            f"smaller than batch_size={args.batch_size}"
        )

    requested_cuda = (
        args.device == "auto"
        or args.device.startswith("cuda")
    )

    pin_memory = (
        requested_cuda
        and torch.cuda.is_available()
    )

    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        seed=args.seed,
    )

    effective_sequences_per_step = (
        args.batch_size
        * args.gradient_accumulation_steps
    )

    effective_tokens_per_step = (
        effective_sequences_per_step
        * seq_len
    )

    print("Stage-A training plan")
    print(
        f"config: {Path(args.config).resolve()}"
    )
    print(
        f"selected shards: {len(shard_paths)}"
    )
    print(
        f"dataset sequences: {len(dataset):,}"
    )
    print(
        f"sequence length: {seq_len}"
    )
    print(
        f"micro batch size: {args.batch_size}"
    )
    print(
        "gradient accumulation: "
        f"{args.gradient_accumulation_steps}"
    )
    print(
        "effective sequences/update: "
        f"{effective_sequences_per_step}"
    )
    print(
        "effective tokens/update: "
        f"{effective_tokens_per_step:,}"
    )
    print(
        f"optimizer updates: {args.max_steps:,}"
    )
    print(
        "planned token consumption: "
        f"{effective_tokens_per_step * args.max_steps:,}"
    )

    if args.dry_run:
        batch = next(
            iter(dataloader)
        )

        print(
            "sample batch shape: "
            f"{tuple(batch['input_ids'].shape)}"
        )
        print(
            "sample batch dtype: "
            f"{batch['input_ids'].dtype}"
        )
        print("Dry run complete.")
        return

    model = CausalLM(
        model_config
    )

    total_params, trainable_params = (
        count_parameters(model)
    )

    print(
        f"parameters: {total_params:,} total | "
        f"{trainable_params:,} trainable"
    )

    optimizer = build_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=args.fused_adamw,
    )

    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        scheduler=scheduler,
        config=TrainerConfig(
            max_steps=args.max_steps,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            max_grad_norm=args.max_grad_norm,
            mixed_precision=args.precision,
            device=args.device,
            log_interval=args.log_interval,
        ),
    )

    summary = trainer.train()

    if not args.no_save:
        checkpoint_path = save_checkpoint(
            output_dir=args.output_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            training_args=args,
            trainer_summary=summary,
        )

        print(
            f"checkpoint saved: {checkpoint_path}"
        )


if __name__ == "__main__":
    main()
