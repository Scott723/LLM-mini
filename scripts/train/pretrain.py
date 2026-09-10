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
DEFAULT_EXPERIMENT_PATH = PROJECT_ROOT / "configs" / "experiment" / "baseline_19m.py"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_model_config(config_path: str | Path):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")

    namespace = runpy.run_path(str(config_path))
    if "MODEL_CONFIG" not in namespace:
        raise KeyError(f"{config_path} must define MODEL_CONFIG")
    return namespace["MODEL_CONFIG"]


def load_experiment(experiment_path: str | Path):
    experiment_path = Path(experiment_path).expanduser().resolve()
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment config file not found: {experiment_path}")

    namespace = runpy.run_path(str(experiment_path))

    if "MODEL_CONFIG_PATH" not in namespace:
        raise KeyError(f"{experiment_path} must define MODEL_CONFIG_PATH")
    if "EXPERIMENT_CONFIG" not in namespace:
        raise KeyError(f"{experiment_path} must define EXPERIMENT_CONFIG")

    settings = dict(namespace["EXPERIMENT_CONFIG"])
    model_config_path = (experiment_path.parent / namespace["MODEL_CONFIG_PATH"]).resolve()
    model_config = load_model_config(model_config_path)

    return model_config, model_config_path, settings


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic pretraining entrypoint.")

    parser.add_argument("--experiment", type=str, default=str(DEFAULT_EXPERIMENT_PATH))

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--val-data-dir", type=str, default=None)
    parser.add_argument("--val-pattern", type=str, default=None)

    parser.add_argument("--eval", dest="eval_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)

    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--attention-backend", choices=["eager", "sdpa"], default=None)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--precision", choices=["fp32", "bf16"], default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--fused-adamw",
        dest="fused_adamw",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save", dest="save_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None)

    parser.add_argument("--wandb", dest="wandb_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default=None)

    parser.add_argument("--device-peak-tflops", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def resolve_settings(cli_args: argparse.Namespace, defaults: dict) -> argparse.Namespace:
    settings = dict(defaults)

    for name, value in vars(cli_args).items():
        if name in {"experiment", "dry_run"}:
            continue
        if value is not None:
            settings[name] = value

    settings["experiment"] = str(Path(cli_args.experiment).expanduser().resolve())
    settings["dry_run"] = cli_args.dry_run

    required = {
        "name",
        "data_dir",
        "pattern",
        "val_data_dir",
        "val_pattern",
        "eval_enabled",
        "eval_interval",
        "eval_max_batches",
        "seq_len",
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "min_lr_ratio",
        "max_grad_norm",
        "precision",
        "device",
        "num_workers",
        "log_interval",
        "seed",
        "fused_adamw",
        "output_dir",
        "save_enabled",
        "save_interval",
        "wandb_enabled",
        "wandb_project",
        "wandb_mode",
    }
    missing = sorted(name for name in required if name not in settings)
    if missing:
        raise KeyError(f"Experiment config is missing required fields: {missing}")

    settings.setdefault("max_shards", None)
    settings.setdefault("resume_from", None)
    settings.setdefault("wandb_run_name", None)
    settings.setdefault("wandb_entity", None)
    settings.setdefault("device_peak_tflops", None)
    settings.setdefault("attention_backend", "eager")
    settings.setdefault("activation_checkpointing", False)

    return argparse.Namespace(**settings)


def validate_settings(args: argparse.Namespace, model_config) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if args.seq_len > model_config.max_position_embeddings:
        raise ValueError(
            f"seq_len={args.seq_len} exceeds "
            f"config.max_position_embeddings={model_config.max_position_embeddings}"
        )
    if args.warmup_steps < 0 or args.warmup_steps > args.max_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps <= max_steps")
    if args.eval_enabled and (args.eval_interval <= 0 or args.eval_max_batches <= 0):
        raise ValueError("eval_interval and eval_max_batches must be positive")
    if args.save_enabled and args.save_interval <= 0:
        raise ValueError("save_interval must be positive")
    if args.device_peak_tflops is not None and args.device_peak_tflops <= 0:
        raise ValueError("device_peak_tflops must be positive or omitted")
    if args.attention_backend not in {"eager", "sdpa"}:
        raise ValueError("attention_backend must be 'eager' or 'sdpa'")


def select_shards(data_dir: str | Path, pattern: str, max_shards: int | None) -> list[Path]:
    paths = find_token_shards(directory=data_dir, pattern=pattern)

    if max_shards is not None:
        if max_shards <= 0:
            raise ValueError("max_shards must be positive or omitted")
        paths = paths[:max_shards]

    if not paths:
        raise RuntimeError("No token shards selected")

    return paths


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def build_run_config(
    model_config,
    model_config_path: Path,
    args: argparse.Namespace,
    train_shard_paths: list[Path],
    val_shard_paths: list[Path],
) -> dict:
    return {
        "experiment": {
            "name": args.name,
            "experiment_file": args.experiment,
            "model_config_file": str(model_config_path),
        },
        "model": to_plain_dict(model_config),
        "training": {
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_tokens_per_step": (
                args.batch_size * args.gradient_accumulation_steps * args.seq_len
            ),
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "max_grad_norm": args.max_grad_norm,
            "precision": args.precision,
            "device": args.device,
            "seed": args.seed,
            "device_peak_tflops": args.device_peak_tflops,
            "attention_backend": args.attention_backend,
            "activation_checkpointing": args.activation_checkpointing,
        },
        "data": {
            "train_data_dir": str(resolve_project_path(args.data_dir)),
            "train_pattern": args.pattern,
            "train_shards": len(train_shard_paths),
            "val_data_dir": (
                str(resolve_project_path(args.val_data_dir)) if args.eval_enabled else None
            ),
            "val_pattern": args.val_pattern if args.eval_enabled else None,
            "val_shards": len(val_shard_paths),
        },
    }


def validate_resume_compatibility(
    checkpoint: dict,
    model_config,
    args: argparse.Namespace,
) -> None:
    if checkpoint["model_config"] != to_plain_dict(model_config):
        raise ValueError("Checkpoint model_config does not match the current model config")

    old_args = checkpoint["training_args"]
    comparisons = {
        "seq_len": (old_args.get("seq_len"), args.seq_len),
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

    mismatches = [
        f"{name}: checkpoint={old_value!r}, current={new_value!r}"
        for name, (old_value, new_value) in comparisons.items()
        if old_value is not None and old_value != new_value
    ]
    if mismatches:
        raise ValueError("Resume configuration mismatch:\n  " + "\n  ".join(mismatches))


def main() -> None:
    cli_args = parse_cli_args()
    model_config, model_config_path, defaults = load_experiment(cli_args.experiment)
    args = resolve_settings(cli_args, defaults)
    validate_settings(args, model_config)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    resume_checkpoint = None
    if args.resume_from is not None:
        resume_checkpoint = load_checkpoint(resolve_project_path(args.resume_from))
        validate_resume_compatibility(resume_checkpoint, model_config, args)

    train_data_dir = resolve_project_path(args.data_dir)
    train_shard_paths = select_shards(train_data_dir, args.pattern, args.max_shards)
    train_dataset = TokenShardDataset(shard_paths=train_shard_paths, seq_len=args.seq_len)

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

    if args.eval_enabled:
        val_data_dir = resolve_project_path(args.val_data_dir)
        val_shard_paths = select_shards(val_data_dir, args.val_pattern, None)
        val_dataset = TokenShardDataset(shard_paths=val_shard_paths, seq_len=args.seq_len)
        eval_dataloader = build_dataloader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            seed=None,
        )

    effective_tokens_per_step = (
        args.batch_size * args.gradient_accumulation_steps * args.seq_len
    )

    print("Pretraining plan")
    print(f"experiment: {args.name}")
    print(f"experiment config: {Path(args.experiment).resolve()}")
    print(f"model config: {model_config_path}")
    print(f"attention backend: {args.attention_backend}")
    print(f"activation checkpointing: {'enabled' if args.activation_checkpointing else 'disabled'}")
    print(f"training shards: {len(train_shard_paths)}")
    print(f"dataset sequences: {len(train_dataset):,}")
    print(f"sequence length: {args.seq_len}")
    print(f"micro batch size: {args.batch_size}")
    print(f"gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"effective tokens/update: {effective_tokens_per_step:,}")
    print(f"optimizer updates: {args.max_steps:,}")
    print(f"planned token consumption: {effective_tokens_per_step * args.max_steps:,}")

    if args.eval_enabled:
        print(f"validation shards: {len(val_shard_paths)}")
        print(f"validation sequences: {len(val_dataset):,}")
        print(f"eval interval: {args.eval_interval} steps")
        print(f"eval max batches: {args.eval_max_batches}")
    else:
        print("validation: disabled")

    print(
        f"checkpoint interval: {args.save_interval} steps"
        if args.save_enabled
        else "checkpointing: disabled"
    )
    print(f"resume from: {args.resume_from or 'none'}")
    print(
        f"wandb: enabled (project={args.wandb_project}, mode={args.wandb_mode})"
        if args.wandb_enabled
        else "wandb: disabled"
    )
    print(
        f"MFU profiling: enabled (peak={args.device_peak_tflops:.2f} TFLOPs)"
        if args.device_peak_tflops is not None
        else "MFU profiling: disabled"
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
    model.set_attention_backend(args.attention_backend)
    model.set_activation_checkpointing(args.activation_checkpointing)
    total_params, trainable_params = count_trainable_parameters(model)
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

    logger = build_logger(
        enable_wandb=args.wandb_enabled,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config=build_run_config(
            model_config=model_config,
            model_config_path=model_config_path,
            args=args,
            train_shard_paths=train_shard_paths,
            val_shard_paths=val_shard_paths,
        ),
    )

    def checkpoint_callback(trainer_state: dict[str, object]) -> None:
        path = save_checkpoint(
            output_dir=resolve_project_path(args.output_dir),
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
            checkpoint_callback=checkpoint_callback if args.save_enabled else None,
            config=TrainerConfig(
                max_steps=args.max_steps,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                mixed_precision=args.precision,
                device=args.device,
                log_interval=args.log_interval,
                eval_interval=args.eval_interval if args.eval_enabled else None,
                eval_max_batches=args.eval_max_batches if args.eval_enabled else None,
                save_interval=args.save_interval if args.save_enabled else None,
                device_peak_tflops=args.device_peak_tflops,
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

        trainer.train()

        if args.save_enabled:
            final_path = save_checkpoint(
                output_dir=resolve_project_path(args.output_dir),
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
