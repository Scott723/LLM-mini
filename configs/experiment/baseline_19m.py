# Architecture is kept separate from the training recipe.
MODEL_CONFIG_PATH = "../model/model_19m.py"

EXPERIMENT_CONFIG = {
    "name": "baseline_19m",

    "data_dir": "data/fineweb_tokenized_v1/train",
    "pattern": "train_*.npy",
    "max_shards": None,

    "val_data_dir": "data/fineweb_tokenized_v1/val",
    "val_pattern": "val_*.npy",
    "eval_enabled": True,
    "eval_interval": 1000,
    "eval_max_batches": 500,

    "seq_len": 512,
    "batch_size": 32,
    "gradient_accumulation_steps": 2,
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

    "output_dir": "checkpoints/baseline_19m",
    "save_enabled": True,
    "save_interval": 10000,
    "resume_from": None,

    "wandb_enabled": True,
    "wandb_project": "llm-mini",
    "wandb_run_name": "baseline-19m",
    "wandb_entity": None,
    "wandb_mode": "online",

    # Hardware-specific. Supply from CLI when the actual GPU allocation is known.
    "device_peak_tflops": None,
}
