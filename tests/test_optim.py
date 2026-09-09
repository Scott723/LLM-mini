import torch
import torch.nn as nn
import torch.nn.functional as F

from myqwen.training import (
    build_lr_scheduler,
    build_optimizer,
    cosine_lr_multiplier,
)


class ToyModel(nn.Module):
    """
    Small model used to test optimizer parameter grouping.

    It contains:
        - Embedding weight      -> decay
        - Linear weight         -> decay
        - Linear bias           -> no decay
        - LayerNorm weight      -> no decay
        - LayerNorm bias        -> no decay
    """

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=32,
            embedding_dim=8,
        )

        self.linear = nn.Linear(
            in_features=8,
            out_features=8,
            bias=True,
        )

        self.norm = nn.LayerNorm(
            normalized_shape=8,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.linear(x)
        x = self.norm(x)

        return x


def test_optimizer_parameter_groups():
    """
    Verify weight-decay grouping:

        ndim >= 2 -> weight decay
        ndim < 2  -> no weight decay
    """

    model = ToyModel()

    weight_decay = 0.1

    optimizer = build_optimizer(
        model=model,
        learning_rate=1e-3,
        weight_decay=weight_decay,
    )

    assert len(optimizer.param_groups) == 2

    decay_group = next(
        group
        for group in optimizer.param_groups
        if group["weight_decay"] == weight_decay
    )

    no_decay_group = next(
        group
        for group in optimizer.param_groups
        if group["weight_decay"] == 0.0
    )

    decay_param_ids = {
        id(parameter)
        for parameter in decay_group["params"]
    }

    no_decay_param_ids = {
        id(parameter)
        for parameter in no_decay_group["params"]
    }

    # Matrix-like parameters should receive weight decay.
    assert id(model.embedding.weight) in decay_param_ids
    assert id(model.linear.weight) in decay_param_ids

    # 1D parameters should not receive weight decay.
    assert id(model.linear.bias) in no_decay_param_ids
    assert id(model.norm.weight) in no_decay_param_ids
    assert id(model.norm.bias) in no_decay_param_ids

    # A parameter must never appear in both groups.
    assert decay_param_ids.isdisjoint(
        no_decay_param_ids
    )

    # Every trainable parameter should appear exactly once.
    all_optimizer_param_ids = (
        decay_param_ids
        | no_decay_param_ids
    )

    all_trainable_param_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    assert (
        all_optimizer_param_ids
        == all_trainable_param_ids
    )


def test_optimizer_step_updates_parameters():
    """
    Verify the basic training sequence:

        forward
        -> loss
        -> backward
        -> optimizer.step()

    actually updates model parameters.
    """

    torch.manual_seed(42)

    model = nn.Linear(
        in_features=4,
        out_features=2,
    )

    optimizer = build_optimizer(
        model=model,
        learning_rate=1e-2,
        weight_decay=0.0,
    )

    x = torch.randn(
        8,
        4,
    )

    target = torch.randn(
        8,
        2,
    )

    weight_before = (
        model.weight
        .detach()
        .clone()
    )

    optimizer.zero_grad()

    prediction = model(x)

    loss = F.mse_loss(
        prediction,
        target,
    )

    loss.backward()

    assert model.weight.grad is not None
    assert torch.isfinite(
        model.weight.grad
    ).all()

    optimizer.step()

    weight_after = (
        model.weight
        .detach()
        .clone()
    )

    assert not torch.equal(
        weight_before,
        weight_after,
    ), "Optimizer step did not update model.weight"


def test_cosine_lr_multiplier():
    """
    Verify the intended warmup + cosine-decay schedule.

    For:
        warmup_steps = 2
        max_steps = 10
        min_lr_ratio = 0.1

    expected landmarks:

        step 0  -> 0.5
        step 1  -> 1.0
        step 2  -> 1.0
        step 10 -> 0.1
    """

    warmup_steps = 2
    max_steps = 10
    min_lr_ratio = 0.1

    multiplier_step_0 = cosine_lr_multiplier(
        step=0,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr_ratio=min_lr_ratio,
    )

    multiplier_step_1 = cosine_lr_multiplier(
        step=1,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr_ratio=min_lr_ratio,
    )

    multiplier_step_2 = cosine_lr_multiplier(
        step=2,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr_ratio=min_lr_ratio,
    )

    multiplier_final = cosine_lr_multiplier(
        step=max_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr_ratio=min_lr_ratio,
    )

    assert abs(
        multiplier_step_0 - 0.5
    ) < 1e-8

    assert abs(
        multiplier_step_1 - 1.0
    ) < 1e-8

    assert abs(
        multiplier_step_2 - 1.0
    ) < 1e-8

    assert abs(
        multiplier_final - min_lr_ratio
    ) < 1e-8

    # After warmup, learning rate should decrease monotonically.
    decay_values = [
        cosine_lr_multiplier(
            step=step,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
        )
        for step in range(
            warmup_steps,
            max_steps + 1,
        )
    ]

    assert all(
        earlier >= later
        for earlier, later
        in zip(
            decay_values,
            decay_values[1:],
        )
    )


def test_lr_scheduler_changes_learning_rate():
    """
    Smoke-test the PyTorch LambdaLR wrapper itself.
    """

    model = nn.Linear(
        in_features=4,
        out_features=2,
    )

    base_lr = 1e-3

    optimizer = build_optimizer(
        model=model,
        learning_rate=base_lr,
        weight_decay=0.0,
    )

    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=2,
        max_steps=10,
        min_lr_ratio=0.1,
    )

    # LambdaLR applies the step-0 multiplier when initialized.
    initial_lr = optimizer.param_groups[0]["lr"]

    assert initial_lr <= base_lr

    optimizer.step()
    scheduler.step()

    lr_after_one_step = (
        optimizer.param_groups[0]["lr"]
    )

    assert lr_after_one_step >= initial_lr


if __name__ == "__main__":
    test_optimizer_parameter_groups()
    print("test_optimizer_parameter_groups: PASSED")

    test_optimizer_step_updates_parameters()
    print("test_optimizer_step_updates_parameters: PASSED")

    test_cosine_lr_multiplier()
    print("test_cosine_lr_multiplier: PASSED")

    test_lr_scheduler_changes_learning_rate()
    print("test_lr_scheduler_changes_learning_rate: PASSED")

    print("\nAll optimizer tests passed.")
