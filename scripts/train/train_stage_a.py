from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import torch

from myqwen.data import TokenShardDataset, build_dataloader, find_token_shards
from myqwen.modeling import CausalLM
from myqwen.training import (
    Trainer,
    TrainerConfig,
    build_logger,
    build_lr_scheduler,
    build_optimizer,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
    to_plain_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config_a.py"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "fineweb_tokenized_v1" / "train"
DEFAULT_VAL_DATA_DIR = PROJECT_ROOT / "data" / "fineweb_tokenized_v1" / "val"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "stage_a"


def load_model_config(config_path: str | Path):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")

    namespace = runpy.run_path(str(config_path))
    if "MODEL_CONFIG" not in namespace:
        raise KeyError(f"{config_path} must define MODEL_CONFIG")
    return namespace["MODEL_CONFIG"]


def select_shards(data_dir: str | Path, pattern: str, max_shards: int | None) -> list[Path]:
    paths = find_token_shards(directory=data_dir, pattern=pattern)
    if max_shards is not None:
        if max_shards <= 0:
            raise ValueError("max_shards must be positive or omitted")
        paths = paths[:max_shards]
    if not paths:
        raise RuntimeError("No token shards selected")
    return paths


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def build_experiment_config(
    model_config,
    args: argparse.Namespace,
    seq_len: int,
    train_shard_paths: list[Path],
    val_shard_paths: list[Path],
) -> dict:
    return {
        "model": to_plain_dict(model_config),
        "training": {
            "seq_len": seq_len,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_tokens_per_step": args.batch_size * args.gradient_accumulation_steps * seq_len,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "max_grad_norm": args.max_grad_norm,
            "precision": args.precision,
            "device": args.device,
            "seed": args.seed,
            "eval_interval": None if args.no_eval else args.eval_interval,
            "eval_max_batches": None if args.no_eval else args.eval_max_batches,
            "save_interval": None if args.no_save else args.save_interval,
            "resume_from": args.resume_from,
        },
        "data": {
            "train_data_dir": str(Path(args.data_dir).resolve()),
            "train_pattern": args.pattern,
            "train_shards": len(train_shard_paths),
            "val_data_dir": None if args.no_eval else str(Path(args.val_data_dir).resolve()),
            "val_pattern": None if args.no_eval else args.val_pattern,
            "val_shards": len(val_shard_paths),
        },
    }


def validate_resume_compatibility(checkpoint: dict, model_config, args: argparse.Namespace, seq_len: int) -> None:
    if checkpoint["model_config"] != to_plain_dict(model_config):
        raise ValueError("Checkpoint model_config does not match the current model config")

    old_args = checkpoint["training_args"]
    old_seq_len = old_args.get("seq_len")
    if old_seq_len is None:
        old_seq_len = checkpoint["model_config"]["max_position_embeddings"]

    comparisons = {
        "seq_len": (old_seq_len, seq_len),
        "batch_size": (old_args.get("batch_size"), args.batch_size),
        "gradient_accumulation_steps": (
            old_args.get("gradient_accumulation_steps"),
            args.gradient_accumulation_steps,
        ),
        "max_steps": (old_args.get("max_steps"), args.max_steps),
        "learning_rate": (old_args.get("learning_rate"), args.learning_rate),
        "weight_decay": (old_args.get("weight_decay"), args.weight_decay),
        "warmup_steps": (old_args.get("warmup_steps"), args.warmup_steps),
        "min_lr_ratio": (old_args.get("min_lr_ratio"), args.min_lr_ratio),
        "max_grad_norm": (old_args.get("max_grad_norm"), args.max_grad_norm),
        "precision": (old_args.get("precision"), args.precision),
        "seed": (old_args.get("seed"), args.seed),
    }

    mismatches = []
    for name, (old_value, new_value) in comparisons.items():
        if old_value is not None and old_value != new_value:
            mismatches.append(f"{name}: checkpoint={old_value!r}, current={new_value!r}")

    if mismatches:
        raise ValueError("Resume configuration mismatch:\n  " + "\n  ".join(mismatches))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-A pretraining for the configurable decoder-only LM.")

    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--pattern", type=str, default="train_*.npy")
    parser.add_argument("--max-shards", type=int, default=None)

    parser.add_argument("--val-data-dir", type=str, default=str(DEFAULT_VAL_DATA_DIR))
    parser.add_argument("--val-pattern", type=str, default="val_*.npy")
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-max-batches", type=int, default=500)
    parser.add_argument("--no-eval", action="store_true")

    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fused-adamw", action="store_true")

    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--no-save", action="store_true")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="llm-mini")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")

    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.warmup_steps < 0 or args.warmup_steps > args.max_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps <= max_steps")
    if not args.no_eval and (args.eval_interval <= 0 or args.eval_max_batches <= 0):
        raise ValueError("eval_interval and eval_max_batches must be positive")
    if not args.no_save and args.save_interval <= 0:
        raise ValueError("save_interval must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_config = load_model_config(args.config)
    seq_len = args.seq_len if args.seq_len is not None else model_config.max_position_embeddings

    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if seq_len > model_config.max_position_embeddings:
        raise ValueError(
            f"seq_len={seq_len} exceeds config.max_position_embeddings={model_config.max_position_embeddings}"
        )

    resume_checkpoint = None
    if args.resume_from is not None:
        resume_checkpoint = load_checkpoint(args.resume_from)
        validate_resume_compatibility(resume_checkpoint, model_config, args, seq_len)

    train_shard_paths = select_shards(args.data_dir, args.pattern, args.max_shards)
    train_dataset = TokenShardDataset(shard_paths=train_shard_paths, seq_len=seq_len)

    if len(train_dataset) < args.batch_size:
        raise ValueError("Training dataset is smaller than batch_size")

    requested_cuda = args.device == "auto" or args.device.startswith("cuda")
    pin_memory = requested_cuda and torch.cuda.is_available()

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        seed=args.seed,
    )

    eval_dataloader = None
    val_shard_paths: list[Path] = []
    val_dataset = None

    if not args.no_eval:
        val_shard_paths = select_shards(args.val_data_dir, args.val_pattern, None)
        val_dataset = TokenShardDataset(shard_paths=val_shard_paths, seq_len=seq_len)
        eval_dataloader = build_dataloader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            seed=None,
        )

    effective_tokens_per_step = args.batch_size * args.gradient_accumulation_steps * seq_len

    print("Stage-A training plan")
    print(f"config: {Path(args.config).resolve()}")
    print(f"training shards: {len(train_shard_paths)}")
    print(f"selected shards: {len(train_shard_paths)}")
    print(f"dataset sequences: {len(train_dataset):,}")
    print(f"sequence length: {seq_len}")
    print(f"micro batch size: {args.batch_size}")
    print(f"gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"effective tokens/update: {effective_tokens_per_step:,}")
    print(f"optimizer updates: {args.max_steps:,}")
    print(f"planned token consumption: {effective_tokens_per_step * args.max_steps:,}")

    if args.no_eval:
        print("validation: disabled")
    else:
        print(f"validation shards: {len(val_shard_paths)}")
        print(f"validation sequences: {len(val_dataset):,}")
        print(f"eval interval: {args.eval_interval} steps")
        print(f"eval max batches: {args.eval_max_batches}")

    print("checkpointing: disabled" if args.no_save else f"checkpoint interval: {args.save_interval} steps")
    print(f"resume from: {args.resume_from or 'none'}")
    print(
        f"wandb: enabled (project={args.wandb_project}, mode={args.wandb_mode})"
        if args.wandb
        else "wandb: disabled"
    )

    if args.dry_run:
        batch = next(iter(train_dataloader))
        print(f"sample batch shape: {tuple(batch['input_ids'].shape)}")
        print(f"sample batch dtype: {batch['input_ids'].dtype}")
        if eval_dataloader is not None:
            val_batch = next(iter(eval_dataloader))
            print(f"validation sample batch shape: {tuple(val_batch['input_ids'].shape)}")
        print("Dry run complete.")
        return

    model = CausalLM(model_config)
    total_params, trainable_params = count_parameters(model)
    print(f"parameters: {total_params:,} total | {trainable_params:,} trainable")

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

    experiment_config = build_experiment_config(
        model_config=model_config,
        args=args,
        seq_len=seq_len,
        train_shard_paths=train_shard_paths,
        val_shard_paths=val_shard_paths,
    )
    logger = build_logger(
        enable_wandb=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config=experiment_config,
    )

    def checkpoint_callback(trainer_state: dict[str, object]) -> None:
        path = save_checkpoint(
            output_dir=args.output_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            training_args=args,
            trainer_state=trainer_state,
        )
        print(f"checkpoint saved: {path}")

    try:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            scheduler=scheduler,
            eval_dataloader=eval_dataloader,
            logger=logger,
            checkpoint_callback=None if args.no_save else checkpoint_callback,
            config=TrainerConfig(
                max_steps=args.max_steps,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                mixed_precision=args.precision,
                device=args.device,
                log_interval=args.log_interval,
                eval_interval=None if args.no_eval else args.eval_interval,
                eval_max_batches=None if args.no_eval else args.eval_max_batches,
                save_interval=None if args.no_save else args.save_interval,
            ),
        )

        if resume_checkpoint is not None:
            restored_state = restore_checkpoint(
                resume_checkpoint,
                model=trainer.model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            trainer.load_state_dict(restored_state)
            print(
                f"resumed training state: step={trainer.global_step}, "
                f"micro_step={trainer.micro_step}, tokens_seen={trainer.tokens_seen:,}"
            )

        summary = trainer.train()

        if not args.no_save:
            final_path = save_checkpoint(
                output_dir=args.output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                model_config=model_config,
                training_args=args,
                trainer_state=trainer.state_dict(),
            )
            print(f"final checkpoint saved: {final_path}")

    finally:
        logger.finish()


if __name__ == "__main__":
    main()
