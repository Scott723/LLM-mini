from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenShardDataset(Dataset):
    """
    Dataset for pre-tokenized causal language-model data.

    Expected file format
    --------------------
    Each shard is a 1D NumPy ``.npy`` array containing one continuous
    token stream, for example:

        [tok_0, tok_1, tok_2, ..., tok_N]

    For our GPT-2 tokenizer pipeline, the on-disk dtype can be uint16
    because vocab_size=50257 < 65536.

    Documents may already be separated by EOT tokens during the
    preprocessing stage. This Dataset does not need to know where
    document boundaries are.

    Sampling strategy
    -----------------
    Each shard is split into non-overlapping chunks of ``seq_len``:

        [0 : seq_len]
        [seq_len : 2 * seq_len]
        ...

    Any trailing tokens shorter than ``seq_len`` are ignored.

    Output
    ------
    Each sample is:

        {
            "input_ids": LongTensor[seq_len]
        }

    Labels are intentionally NOT created here. For causal LM training,
    ``input_ids`` can also be passed as ``labels`` to
    ``compute_causal_lm_loss``, which performs the next-token shift.
    """

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        seq_len: int,
    ):
        super().__init__()

        if seq_len < 2:
            raise ValueError(
                f"seq_len must be at least 2, got {seq_len}"
            )

        if len(shard_paths) == 0:
            raise ValueError(
                "shard_paths cannot be empty"
            )

        self.seq_len = seq_len

        self.shard_paths = [
            Path(path).expanduser().resolve()
            for path in shard_paths
        ]

        self.shards: list[np.ndarray] = []
        self.tokens_per_shard: list[int] = []
        self.sequences_per_shard: list[int] = []

        # cumulative_sequence_counts[i] is the global sequence index
        # at which shard i starts.
        #
        # Example:
        #   sequences_per_shard = [10, 20, 5]
        #
        #   cumulative = [0, 10, 30, 35]
        self.cumulative_sequence_counts = [0]

        total_sequences = 0

        for path in self.shard_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"Token shard not found: {path}"
                )

            if path.suffix != ".npy":
                raise ValueError(
                    f"Expected a .npy token shard, got: {path}"
                )

            # Memory-map the file instead of loading the whole shard
            # into RAM.
            shard = np.load(
                path,
                mmap_mode="r",
            )

            if shard.ndim != 1:
                raise ValueError(
                    f"Token shard must be 1D, but {path} has "
                    f"shape {shard.shape}"
                )

            if not np.issubdtype(
                shard.dtype,
                np.integer,
            ):
                raise TypeError(
                    f"Token shard must contain integer token ids, "
                    f"but {path} has dtype {shard.dtype}"
                )

            num_tokens = int(shard.shape[0])
            num_sequences = (
                num_tokens // self.seq_len
            )

            self.shards.append(shard)
            self.tokens_per_shard.append(
                num_tokens
            )
            self.sequences_per_shard.append(
                num_sequences
            )

            total_sequences += num_sequences
            self.cumulative_sequence_counts.append(
                total_sequences
            )

        self.total_sequences = total_sequences
        self.total_tokens = sum(
            self.tokens_per_shard
        )

        if self.total_sequences == 0:
            raise ValueError(
                "No complete training sequence can be formed. "
                f"All shards contain fewer than seq_len={seq_len} "
                "usable tokens."
            )

    def __len__(self) -> int:
        return self.total_sequences

    def _locate_sequence(
        self,
        index: int,
    ) -> tuple[int, int]:
        """
        Map a global Dataset index to:

            (shard_index, local_sequence_index)
        """

        if index < 0:
            index += len(self)

        if not (
            0 <= index < len(self)
        ):
            raise IndexError(
                f"Dataset index out of range: {index}"
            )

        # Find the rightmost cumulative boundary <= index.
        shard_index = (
            bisect_right(
                self.cumulative_sequence_counts,
                index,
            )
            - 1
        )

        local_sequence_index = (
            index
            - self.cumulative_sequence_counts[
                shard_index
            ]
        )

        return (
            shard_index,
            local_sequence_index,
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        shard_index, local_index = (
            self._locate_sequence(index)
        )

        shard = self.shards[
            shard_index
        ]

        start = (
            local_index
            * self.seq_len
        )

        end = (
            start
            + self.seq_len
        )

        token_chunk = shard[
            start:end
        ]

        # np.load(..., mmap_mode="r") returns a read-only memory-mapped
        # array. torch.tensor() intentionally copies the current chunk
        # into a writable int64 tensor, which is the dtype required by
        # nn.Embedding.
        input_ids = torch.tensor(
            token_chunk,
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
        }

    def summary(
        self,
    ) -> dict[str, object]:
        """
        Return simple Dataset metadata useful for debugging/logging.
        """

        return {
            "num_shards": len(
                self.shards
            ),
            "seq_len": self.seq_len,
            "total_tokens_on_disk": (
                self.total_tokens
            ),
            "total_sequences": (
                self.total_sequences
            ),
            "tokens_per_shard": list(
                self.tokens_per_shard
            ),
            "sequences_per_shard": list(
                self.sequences_per_shard
            ),
        }


def find_token_shards(
    directory: str | Path,
    pattern: str = "*.npy",
) -> list[Path]:
    """
    Find and sort tokenized .npy shards in a directory.

    Example:

        paths = find_token_shards(
            "data/fineweb_tokenized",
            pattern="train_*.npy",
        )
    """

    directory = (
        Path(directory)
        .expanduser()
        .resolve()
    )

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {directory}"
        )

    paths = sorted(
        directory.glob(pattern)
    )

    if len(paths) == 0:
        raise FileNotFoundError(
            f"No token shards matching '{pattern}' "
            f"were found in {directory}"
        )

    return paths


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
    seed: int | None = None,
) -> DataLoader:
    """
    Build a PyTorch DataLoader for LM pretraining.

    Args:
        dataset:
            Usually a TokenShardDataset.

        batch_size:
            Number of sequences per batch.

        shuffle:
            Shuffle sequence indices every epoch.

        num_workers:
            Number of DataLoader worker processes.

            Start with 0 while debugging. Increase later when the
            training pipeline is stable.

        pin_memory:
            Pin CPU tensors for faster CPU -> CUDA transfer.

            Usually enable this for GPU training.

        drop_last:
            Drop the final incomplete batch.

            Usually True for pretraining so every optimization step
            has a stable batch shape.

        seed:
            Optional seed for reproducible DataLoader shuffling.

    Returns:
        torch.utils.data.DataLoader
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers must be >= 0"
        )

    generator = None

    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(
            seed
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(
            num_workers > 0
        ),
        generator=generator,
    )
