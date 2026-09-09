import torch
import torch.nn.functional as F


def compute_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute next-token cross-entropy loss for a causal language model.

    Args:
        logits:
            Model output with shape:

                [B, T, vocab_size]

        labels:
            Target token ids with shape:

                [B, T]

            For standard pretraining, labels can simply be the same
            tensor as input_ids. This function performs the one-token
            shift internally.

        ignore_index:
            Label value that should not contribute to the loss.

            Default:
                -100

            This is useful later for SFT, where some positions such as
            prompt tokens may be masked out from the training loss.

    Returns:
        loss:
            Scalar tensor.

    Example:

        tokens:
            [A, B, C, D, E]

        The model output at:

            A -> predicts B
            B -> predicts C
            C -> predicts D
            D -> predicts E

        Therefore:

            logits[:, :-1, :]
            aligns with
            labels[:, 1:]
    """

    if logits.dim() != 3:
        raise ValueError(
            "logits must have shape [B, T, vocab_size], "
            f"got {tuple(logits.shape)}"
        )

    if labels.dim() != 2:
        raise ValueError(
            "labels must have shape [B, T], "
            f"got {tuple(labels.shape)}"
        )

    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "The first two dimensions of logits must match labels. "
            f"Got logits shape {tuple(logits.shape)} and "
            f"labels shape {tuple(labels.shape)}"
        )

    if logits.shape[1] < 2:
        raise ValueError(
            "Sequence length must be at least 2 to compute "
            "next-token prediction loss"
        )

    # ----------------------------------------------------------
    # Shift for next-token prediction.
    #
    # logits at position t predict token t + 1.
    #
    # logits:
    #     [B, T, V]
    #
    # shift_logits:
    #     [B, T - 1, V]
    #
    # labels:
    #     [B, T]
    #
    # shift_labels:
    #     [B, T - 1]
    # ----------------------------------------------------------
    shift_logits = (
        logits[:, :-1, :]
        .contiguous()
    )

    shift_labels = (
        labels[:, 1:]
        .contiguous()
    )

    vocab_size = logits.shape[-1]

    # ----------------------------------------------------------
    # Flatten batch and sequence dimensions.
    #
    # [B, T - 1, V]
    # ->
    # [B * (T - 1), V]
    #
    # [B, T - 1]
    # ->
    # [B * (T - 1)]
    # ----------------------------------------------------------
    shift_logits = shift_logits.view(
        -1,
        vocab_size,
    )

    shift_labels = shift_labels.view(
        -1
    )

    # ----------------------------------------------------------
    # Cross entropy:
    #
    # CE(logits, target)
    #
    # internally performs:
    #
    #     log_softmax + negative log likelihood
    #
    # so do NOT apply softmax to logits beforehand.
    # ----------------------------------------------------------
    loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=ignore_index,
    )

    return loss
