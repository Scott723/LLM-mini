import math

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool = False,
) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer for language-model training.

    Parameter grouping rule:

        ndim >= 2
            -> apply weight decay

        ndim < 2
            -> no weight decay

    This means matrix-like parameters such as Linear / Embedding
    weights are decayed, while 1D parameters such as normalization
    scales and biases are not.

    Args:
        model:
            Model whose trainable parameters will be optimized.

        learning_rate:
            Base learning rate.

        weight_decay:
            AdamW decoupled weight-decay coefficient.

        betas:
            Adam momentum coefficients.

            Common LLM choice:
                (0.9, 0.95)

        eps:
            Numerical-stability constant used by AdamW.

        fused:
            Whether to request PyTorch fused AdamW.

            Keep False for the reference / learning implementation.
            It can be enabled later on supported CUDA devices.

    Returns:
        torch.optim.AdamW instance.
    """

    decay_params = []
    no_decay_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay_params.append(parameter)
        else:
            no_decay_params.append(parameter)

    if len(decay_params) == 0:
        raise ValueError(
            "No trainable parameters were found for the decay group"
        )

    parameter_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "lr": learning_rate,
        "betas": betas,
        "eps": eps,
    }

    if fused:
        optimizer_kwargs["fused"] = True

    optimizer = torch.optim.AdamW(
        parameter_groups,
        **optimizer_kwargs,
    )

    return optimizer


def cosine_lr_multiplier(
    step: int,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """
    Learning-rate multiplier for linear warmup + cosine decay.

    Schedule:

        0 -------- warmup_steps ---------------- max_steps

        linear warmup
            0 -> 1

        cosine decay
            1 -> min_lr_ratio

    The returned value multiplies the optimizer's base learning rate.
    """

    if warmup_steps < 0:
        raise ValueError(
            "warmup_steps must be >= 0"
        )

    if max_steps <= 0:
        raise ValueError(
            "max_steps must be > 0"
        )

    if warmup_steps >= max_steps:
        raise ValueError(
            "warmup_steps must be smaller than max_steps"
        )

    if not (
        0.0 <= min_lr_ratio <= 1.0
    ):
        raise ValueError(
            "min_lr_ratio must be in [0, 1]"
        )

    if step < warmup_steps:
        if warmup_steps == 0:
            return 1.0

        return (
            float(step + 1)
            / float(warmup_steps)
        )

    if step >= max_steps:
        return min_lr_ratio

    decay_steps = (
        max_steps - warmup_steps
    )

    progress = (
        step - warmup_steps
    ) / decay_steps

    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )

    multiplier = (
        min_lr_ratio
        + (1.0 - min_lr_ratio)
        * cosine
    )

    return multiplier


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Build a PyTorch scheduler implementing:

        linear warmup
            +
        cosine learning-rate decay

    Usage:

        optimizer.step()
        scheduler.step()

    The scheduler only controls learning rate.
    Gradient clipping and gradient accumulation belong in trainer.py.
    """

    def lr_lambda(
        step: int,
    ) -> float:
        return cosine_lr_multiplier(
            step=step,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )
