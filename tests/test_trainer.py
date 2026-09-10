import torch
from torch.utils.data import DataLoader, Dataset

from myqwen.config import ModelConfig
from myqwen.modeling import CausalLM
from myqwen.training import (
    build_optimizer,
    Trainer,
    TrainerConfig,
)


class TinyTokenDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 8,
        seq_len: int = 8,
        vocab_size: int = 64,
    ):
        generator = torch.Generator()
        generator.manual_seed(1234)

        self.input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(
                num_samples,
                seq_len,
            ),
            generator=generator,
            dtype=torch.long,
        )

    def __len__(
        self,
    ) -> int:
        return self.input_ids.shape[
            0
        ]

    def __getitem__(
        self,
        index: int,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        return {
            "input_ids": (
                self.input_ids[
                    index
                ]
            ),
        }


class RecordingLogger:
    def __init__(
        self,
    ):
        self.records = []
        self.finished = False

    def log(
        self,
        metrics,
        step,
    ):
        self.records.append(
            {
                "step": step,
                "metrics": dict(
                    metrics
                ),
            }
        )

    def finish(
        self,
    ):
        self.finished = True


def build_tiny_model(
) -> CausalLM:
    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        qk_norm=False,
        attention_output_gate=False,
        attention_bias=False,
        attention_dropout=0.0,
        position_embedding_type="rope",
        rope_theta=10_000.0,
        partial_rotary_factor=1.0,
        max_position_embeddings=16,
        norm_type="rmsnorm",
        norm_eps=1e-6,
        mlp_type="swiglu",
        hidden_act="silu",
        mlp_bias=False,
        mixer_pattern=(
            "attention",
        ),
        tie_word_embeddings=True,
        initializer_range=0.02,
    )

    return CausalLM(
        config
    )


def build_tiny_dataloader(
    batch_size: int = 2,
) -> DataLoader:
    dataset = TinyTokenDataset(
        num_samples=8,
        seq_len=8,
        vocab_size=64,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )


def test_trainer_updates_model_parameters():
    torch.manual_seed(
        0
    )

    model = (
        build_tiny_model()
    )

    optimizer = (
        build_optimizer(
            model=model,
            learning_rate=1e-3,
            weight_decay=0.0,
            fused=False,
        )
    )

    dataloader = (
        build_tiny_dataloader(
            batch_size=2,
        )
    )

    before = {
        name: (
            parameter
            .detach()
            .clone()
        )
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    }

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        config=TrainerConfig(
            max_steps=3,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            mixed_precision="fp32",
            device="cpu",
            log_interval=1,
        ),
    )

    summary = (
        trainer.train()
    )

    changed = []

    for (
        name,
        parameter,
    ) in model.named_parameters():
        if not parameter.requires_grad:
            continue

        changed.append(
            not torch.equal(
                before[
                    name
                ],
                parameter.detach(),
            )
        )

    assert any(
        changed
    )

    assert (
        trainer.global_step
        == 3
    )

    assert (
        trainer.micro_step
        == 3
    )

    assert (
        trainer.tokens_seen
        == 48
    )

    assert (
        summary[
            "global_step"
        ]
        == 3
    )

    assert (
        summary[
            "validation_history"
        ]
        == []
    )


def test_gradient_accumulation_step_count():
    torch.manual_seed(
        0
    )

    model = (
        build_tiny_model()
    )

    optimizer = (
        build_optimizer(
            model=model,
            learning_rate=1e-3,
            weight_decay=0.0,
            fused=False,
        )
    )

    dataloader = (
        build_tiny_dataloader(
            batch_size=2,
        )
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        config=TrainerConfig(
            max_steps=2,
            gradient_accumulation_steps=3,
            max_grad_norm=1.0,
            mixed_precision="fp32",
            device="cpu",
            log_interval=1,
        ),
    )

    trainer.train()

    assert (
        trainer.global_step
        == 2
    )

    assert (
        trainer.micro_step
        == 6
    )

    assert (
        trainer.tokens_seen
        == 96
    )


def test_trainer_can_cycle_dataloader():
    torch.manual_seed(
        0
    )

    model = (
        build_tiny_model()
    )

    optimizer = (
        build_optimizer(
            model=model,
            learning_rate=1e-3,
            weight_decay=0.0,
            fused=False,
        )
    )

    dataloader = (
        build_tiny_dataloader(
            batch_size=2,
        )
    )

    assert (
        len(
            dataloader
        )
        == 4
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        config=TrainerConfig(
            max_steps=6,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            mixed_precision="fp32",
            device="cpu",
            log_interval=3,
        ),
    )

    trainer.train()

    assert (
        trainer.global_step
        == 6
    )

    assert (
        trainer.micro_step
        == 6
    )

    assert (
        trainer.tokens_seen
        == 96
    )


def test_trainer_runs_validation_at_interval():
    torch.manual_seed(
        0
    )

    model = (
        build_tiny_model()
    )

    optimizer = (
        build_optimizer(
            model=model,
            learning_rate=1e-3,
            weight_decay=0.0,
            fused=False,
        )
    )

    train_dataloader = (
        build_tiny_dataloader(
            batch_size=2,
        )
    )

    eval_dataloader = (
        build_tiny_dataloader(
            batch_size=2,
        )
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=TrainerConfig(
            max_steps=4,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            mixed_precision="fp32",
            device="cpu",
            log_interval=10,
            eval_interval=2,
            eval_max_batches=1,
        ),
    )

    summary = (
        trainer.train()
    )

    history = summary[
        "validation_history"
    ]

    assert len(
        history
    ) == 2

    assert [
        record[
            "step"
        ]
        for record
        in history
    ] == [
        2,
        4,
    ]

    assert (
        model.training
        is True
    )


def test_trainer_requires_eval_dataloader():
    model = (
        build_tiny_model()
    )

    optimizer = (
        build_optimizer(
            model=model,
            learning_rate=1e-3,
            weight_decay=0.0,
            fused=False,
        )
    )

    train_dataloader = (
        build_tiny_dataloader()
    )

    try:
        Trainer(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            eval_dataloader=None,
            config=TrainerConfig(
                max_steps=2,
                mixed_precision="fp32",
                device="cpu",
                eval_interval=1,
                eval_max_batches=1,
            ),
        )
    except ValueError as error:
        assert (
            "eval_dataloader is None"
            in str(
                error
            )
        )
    else:
        raise AssertionError(
            "Expected ValueError when "
            "validation is enabled without "
            "eval_dataloader"
        )


def test_trainer_logs_train_and_validation_metrics():
    torch.manual_seed(
        0
    )

    model = (
        build_tiny_model()
    )

    optimizer = (
        build_optimizer(
            model=model,
            learning_rate=1e-3,
            weight_decay=0.0,
            fused=False,
        )
    )

    train_dataloader = (
        build_tiny_dataloader(
            batch_size=2,
        )
    )

    eval_dataloader = (
        build_tiny_dataloader(
            batch_size=2,
        )
    )

    logger = (
        RecordingLogger()
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        logger=logger,
        config=TrainerConfig(
            max_steps=2,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            mixed_precision="fp32",
            device="cpu",
            log_interval=1,
            eval_interval=2,
            eval_max_batches=1,
        ),
    )

    trainer.train()

    train_records = [
        record
        for record
        in logger.records
        if (
            "train/loss"
            in record[
                "metrics"
            ]
        )
    ]

    val_records = [
        record
        for record
        in logger.records
        if (
            "val/loss"
            in record[
                "metrics"
            ]
        )
    ]

    assert len(
        train_records
    ) == 2

    assert [
        record[
            "step"
        ]
        for record
        in train_records
    ] == [
        1,
        2,
    ]

    assert len(
        val_records
    ) == 1

    assert (
        val_records[
            0
        ][
            "step"
        ]
        == 2
    )

    assert (
        "train/lr"
        in train_records[
            0
        ][
            "metrics"
        ]
    )

    assert (
        "train/grad_norm"
        in train_records[
            0
        ][
            "metrics"
        ]
    )

    assert (
        "val/perplexity"
        in val_records[
            0
        ][
            "metrics"
        ]
    )
