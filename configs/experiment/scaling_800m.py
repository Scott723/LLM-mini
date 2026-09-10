MODEL_CONFIG_PATH = "../model/model_800m.py"

EXPERIMENT_CONFIG = {
    "name": "scaling_800m",

    "data_dir": "data/fineweb_tokenized_v1/train",
    "pattern": "train_*.npy",
    "max_shards": None,

    "val_data_dir": "data/fineweb_tokenized_v1/val",
    "val_pattern": "val_*.npy",
    "eval_enabled": True,
    "eval_interval": 1000,
    "eval_max_batches": 500,

    "seq_len": 512,

    # Initial conservative runtime defaults.
    # 2 x 32 = 64 sequences/update, matching the other scaling runs.
    # Re-benchmark after SDPA and activation checkpointing are available.
    "batch_size": 2,
    "gradient_accumulation_steps": 32,
    "max_steps": 100000,

    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 3000,
    "min_lr_ratio": 0.1,
    "max_grad_norm": 1.0,

    "precision": "bf16",
    "device": "auto",
    "num_workers": 2,
    "log_interval": 10,
    "seed": 42,
    "fused_adamw": False,

    "output_dir": "checkpoints/scaling_800m",
    "save_enabled": True,
    "save_interval": 10000,
    "resume_from": None,

    "wandb_enabled": True,
    "wandb_project": "llm-mini",
    "wandb_run_name": "scaling-800m",
    "wandb_entity": None,
    "wandb_mode": "online",

    "device_peak_tflops": None,
}
