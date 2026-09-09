from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Dataset

from myqwen.config import ModelConfig
from myqwen.modeling import CausalLM
from myqwen.training import compute_causal_lm_loss
from myqwen.training.evaluator import EvaluatorConfig, evaluate_causal_lm


class TinyDataset(Dataset):
    def __init__(self):
        self.samples = [
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            torch.tensor([5, 4, 3, 2, 1], dtype=torch.long),
            torch.tensor([1, 3, 5, 7, 9], dtype=torch.long),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return {"input_ids": self.samples[index]}


def build_model() -> CausalLM:
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=16,
        attention_dropout=0.0,
    )
    torch.manual_seed(123)
    return CausalLM(config)


def test_evaluator_matches_manual_token_weighted_loss():
    model = build_model()
    dataset = TinyDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )

    model.eval()

    expected_nll = 0.0
    expected_tokens = 0

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            logits = model(input_ids)

            loss = compute_causal_lm_loss(
                logits=logits,
                labels=input_ids,
            )

            prediction_tokens = (
                input_ids.shape[0]
                * (input_ids.shape[1] - 1)
            )

            expected_nll += loss.item() * prediction_tokens
            expected_tokens += prediction_tokens

    expected_loss = expected_nll / expected_tokens

    result = evaluate_causal_lm(
        model=model,
        dataloader=dataloader,
        config=EvaluatorConfig(
            device="cpu",
            mixed_precision="fp32",
        ),
    )

    assert math.isclose(
        result["loss"],
        expected_loss,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    assert math.isclose(
        result["perplexity"],
        math.exp(expected_loss),
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    assert result["prediction_tokens"] == expected_tokens
    assert result["num_sequences"] == len(dataset)


def test_evaluator_restores_training_mode():
    model = build_model()
    model.train()

    dataloader = DataLoader(
        TinyDataset(),
        batch_size=2,
        shuffle=False,
    )

    evaluate_causal_lm(
        model=model,
        dataloader=dataloader,
        config=EvaluatorConfig(
            device="cpu",
            mixed_precision="fp32",
            max_batches=1,
        ),
    )

    assert model.training is True
