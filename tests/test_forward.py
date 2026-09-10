import torch

from configs.model.model_19m import MODEL_CONFIG
from myqwen.modeling import CausalLM


def test_model_forward():
    torch.manual_seed(42)

    config = MODEL_CONFIG
    batch_size = 2
    seq_len = 32

    model = CausalLM(config)
    model.eval()

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
    )

    with torch.no_grad():
        logits = model(input_ids=input_ids)

    expected_shape = (batch_size, seq_len, config.vocab_size)

    assert logits.shape == expected_shape, (
        f"Expected logits shape {expected_shape}, got {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"


def test_weight_tying():
    config = MODEL_CONFIG
    model = CausalLM(config)

    if config.tie_word_embeddings:
        assert model.lm_head.weight is model.model.token_embedding.weight


def test_decoder_layer_count():
    config = MODEL_CONFIG
    model = CausalLM(config)

    assert len(model.model.layers) == config.num_hidden_layers
