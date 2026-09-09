import torch

from configs.config_a import MODEL_CONFIG
from myqwen.model import CausalLM


def test_model_forward():
    """
    Basic end-to-end forward test.

    Verifies:
        1. input_ids -> logits can run end to end
        2. output shape is correct
        3. logits contain no NaN / Inf
    """

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
        logits = model(
            input_ids=input_ids,
        )

    expected_shape = (
        batch_size,
        seq_len,
        config.vocab_size,
    )

    assert logits.shape == expected_shape, (
        f"Expected logits shape {expected_shape}, "
        f"got {tuple(logits.shape)}"
    )

    assert torch.isfinite(logits).all(), (
        "Logits contain NaN or Inf"
    )


def test_weight_tying():
    """
    Verify token embedding and LM head share the same Parameter
    when tie_word_embeddings=True.
    """

    config = MODEL_CONFIG

    model = CausalLM(config)

    if config.tie_word_embeddings:
        assert (
            model.lm_head.weight
            is model.model.token_embedding.weight
        ), (
            "LM head weight is not tied to token embedding weight"
        )


def test_decoder_layer_count():
    """
    Verify the model contains the configured number of decoder blocks.
    """

    config = MODEL_CONFIG

    model = CausalLM(config)

    assert (
        len(model.model.layers)
        == config.num_hidden_layers
    ), (
        f"Expected {config.num_hidden_layers} decoder layers, "
        f"got {len(model.model.layers)}"
    )


if __name__ == "__main__":
    test_model_forward()
    print("test_model_forward: PASSED")

    test_weight_tying()
    print("test_weight_tying: PASSED")

    test_decoder_layer_count()
    print("test_decoder_layer_count: PASSED")

    print("\nAll model tests passed.")
