import torch
from torch.utils.data import DataLoader, Dataset

from myqwen.config import ModelConfig
from myqwen.modeling import CausalLM
from myqwen.training import Trainer, TrainerConfig


class TinyTokenDataset(Dataset):
    def __init__(self):
        self.samples = [
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long),
            torch.tensor([2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long),
            torch.tensor([3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.long),
            torch.tensor([4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.long),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {"input_ids": self.samples[index]}


class RecordingLogger:
    def __init__(self):
        self.records = []

    def log(self, metrics, step):
        self.records.append((step, dict(metrics)))

    def finish(self):
        pass


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        position_embedding_type="rope",
        max_position_embeddings=8,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        mixer_pattern=("attention",),
        tie_word_embeddings=True,
    )


def test_trainer_logs_mfu_when_peak_tflops_is_provided():
    torch.manual_seed(0)
    model = CausalLM(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = DataLoader(TinyTokenDataset(), batch_size=2, shuffle=False)
    logger = RecordingLogger()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        logger=logger,
        config=TrainerConfig(
            max_steps=2,
            mixed_precision="fp32",
            device="cpu",
            log_interval=1,
            device_peak_tflops=100.0,
        ),
    )
    summary = trainer.train()

    train_records = [metrics for _, metrics in logger.records if "train/loss" in metrics]

    assert len(train_records) == 2
    assert all("train/achieved_tflops" in metrics for metrics in train_records)
    assert all("train/mfu" in metrics for metrics in train_records)
    assert all(metrics["train/mfu"] >= 0 for metrics in train_records)
    assert "average_mfu" in summary
    assert "average_achieved_tflops" in summary


def test_trainer_config_rejects_invalid_peak_tflops():
    try:
        TrainerConfig(device_peak_tflops=0.0)
    except ValueError as error:
        assert "device_peak_tflops" in str(error)
    else:
        raise AssertionError("Expected invalid device_peak_tflops to raise ValueError")
