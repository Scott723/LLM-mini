from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from myqwen.config import ModelConfig
from myqwen.modeling import CausalLM
from myqwen.training import Trainer, TrainerConfig, build_optimizer


class TinyDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(123)
        self.tokens = torch.randint(0, 32, (8, 8), generator=generator)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return {"input_ids": self.tokens[index]}


def build_model():
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
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
        mixer_pattern=("attention",),
        tie_word_embeddings=True,
        initializer_range=0.02,
    )
    return CausalLM(config)


def build_loader():
    return DataLoader(TinyDataset(), batch_size=2, shuffle=False, drop_last=True)


def test_trainer_state_can_resume_counters():
    model = build_model()
    optimizer = build_optimizer(model=model, learning_rate=1e-3, weight_decay=0.0, fused=False)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=build_loader(),
        config=TrainerConfig(max_steps=1, mixed_precision="fp32", device="cpu", log_interval=1),
    )
    first_summary = trainer.train()

    resumed_model = build_model()
    resumed_optimizer = build_optimizer(
        model=resumed_model, learning_rate=1e-3, weight_decay=0.0, fused=False
    )
    resumed_trainer = Trainer(
        model=resumed_model,
        optimizer=resumed_optimizer,
        dataloader=build_loader(),
        config=TrainerConfig(max_steps=3, mixed_precision="fp32", device="cpu", log_interval=1),
    )
    resumed_trainer.load_state_dict(first_summary)
    resumed_trainer.train()

    assert resumed_trainer.global_step == 3
    assert resumed_trainer.micro_step == 3
    assert resumed_trainer.tokens_seen == 48


def test_periodic_checkpoint_callback_runs_at_expected_steps():
    model = build_model()
    optimizer = build_optimizer(model=model, learning_rate=1e-3, weight_decay=0.0, fused=False)
    saved_steps = []

    def checkpoint_callback(state):
        saved_steps.append(state["global_step"])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=build_loader(),
        checkpoint_callback=checkpoint_callback,
        config=TrainerConfig(
            max_steps=5,
            mixed_precision="fp32",
            device="cpu",
            log_interval=10,
            save_interval=2,
        ),
    )
    trainer.train()

    assert saved_steps == [2, 4]
