import torch

from myqwen.config import ModelConfig
from myqwen.training import compute_causal_lm_loss
from myqwen.modeling import CausalLM


def build_test_config() -> ModelConfig:
    """
    Small model config used only for unit tests.
    """
    return ModelConfig(
        vocab_size=128,

        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,

        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,

        qk_norm=False,
        attention_output_gate=False,
        attention_bias=False,
        attention_dropout=0.0,

        position_embedding_type="rope",
        rope_theta=10_000.0,
        partial_rotary_factor=1.0,
        max_position_embeddings=64,

        norm_type="rmsnorm",
        norm_eps=1e-6,

        mlp_type="swiglu",
        hidden_act="silu",
        mlp_bias=False,

        mixer_pattern=("attention",),

        tie_word_embeddings=True,
        initializer_range=0.02,
    )


def test_causal_lm_loss_forward_backward():
    """
    End-to-end training-graph smoke test:

        input_ids
            -> model
            -> logits
            -> causal LM loss
            -> backward

    Verifies:
        1. loss is a scalar
        2. loss is finite
        3. backward runs successfully
        4. trainable parameters receive finite gradients
    """

    torch.manual_seed(42)

    config = build_test_config()
    model = CausalLM(config)
    model.train()

    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
    )

    logits = model(
        input_ids=input_ids,
    )

    loss = compute_causal_lm_loss(
        logits=logits,
        labels=input_ids,
    )

    # ----------------------------------------------------------
    # 1. Loss should be a scalar tensor.
    # ----------------------------------------------------------
    assert loss.ndim == 0, (
        f"Expected scalar loss, got shape {tuple(loss.shape)}"
    )

    # ----------------------------------------------------------
    # 2. Loss should not be NaN / Inf.
    # ----------------------------------------------------------
    assert torch.isfinite(loss), (
        f"Loss is not finite: {loss.item()}"
    )

    # ----------------------------------------------------------
    # 3. Backward should run without error.
    # ----------------------------------------------------------
    loss.backward()

    # ----------------------------------------------------------
    # 4. Check gradients.
    # ----------------------------------------------------------
    params_with_grad = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.grad is None:
            continue

        params_with_grad += 1

        assert torch.isfinite(parameter.grad).all(), (
            f"Gradient contains NaN / Inf: {name}"
        )

    assert params_with_grad > 0, (
        "No trainable parameter received gradients"
    )


def test_ignore_index():
    """
    Verify that labels equal to ignore_index are accepted and excluded
    from the cross-entropy calculation.

    This will be useful later for SFT prompt masking.
    """

    torch.manual_seed(42)

    config = build_test_config()
    model = CausalLM(config)
    model.eval()

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 16),
        dtype=torch.long,
    )

    labels = input_ids.clone()

    # Mask a few target positions.
    #
    # Remember:
    # compute_causal_lm_loss internally uses labels[:, 1:].
    labels[:, 5:8] = -100

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
        )

        loss = compute_causal_lm_loss(
            logits=logits,
            labels=labels,
            ignore_index=-100,
        )

    assert loss.ndim == 0
    assert torch.isfinite(loss), (
        f"Masked loss is not finite: {loss.item()}"
    )


if __name__ == "__main__":
    test_causal_lm_loss_forward_backward()
    print("test_causal_lm_loss_forward_backward: PASSED")

    test_ignore_index()
    print("test_ignore_index: PASSED")

    print("\nAll loss tests passed.")
