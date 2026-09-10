from __future__ import annotations

from pathlib import Path

import torch

from myqwen.training.checkpoint import load_checkpoint, restore_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path: Path):
    torch.manual_seed(0)

    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    x = torch.randn(2, 4)
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    expected_weight = model.weight.detach().clone()
    expected_bias = model.bias.detach().clone()
    expected_scheduler_epoch = scheduler.last_epoch

    trainer_state = {
        "global_step": 7,
        "micro_step": 14,
        "tokens_seen": 7168,
        "validation_history": [{"step": 5, "loss": 1.2}],
    }

    path = save_checkpoint(
        output_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        model_config={"hidden_size": 4},
        training_args={"max_steps": 100},
        trainer_state=trainer_state,
    )

    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    checkpoint = load_checkpoint(path)
    restored_state = restore_checkpoint(checkpoint, model, optimizer, scheduler)

    assert path.name == "step_000007.pt"
    assert torch.equal(model.weight, expected_weight)
    assert torch.equal(model.bias, expected_bias)
    assert scheduler.last_epoch == expected_scheduler_epoch
    assert restored_state == trainer_state


def test_load_checkpoint_supports_legacy_trainer_summary(tmp_path: Path):
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": None,
            "model_config": {},
            "training_args": {},
            "trainer_summary": {"global_step": 3, "micro_step": 3, "tokens_seen": 24},
        },
        path,
    )

    checkpoint = load_checkpoint(path)
    assert checkpoint["trainer_state"]["global_step"] == 3
