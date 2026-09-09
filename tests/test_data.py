from pathlib import Path
import tempfile

import numpy as np
import torch

from myqwen.data import (
    TokenShardDataset,
    build_dataloader,
    find_token_shards,
)


def create_test_shards(
    directory: Path,
) -> list[Path]:
    """
    Create two tiny token shards for testing.

    seq_len will be 4.

    shard 0:
        10 tokens -> 2 complete sequences
        [0,1,2,3]
        [4,5,6,7]
        trailing [8,9] is ignored

    shard 1:
        8 tokens -> 2 complete sequences
        [100,101,102,103]
        [104,105,106,107]
    """

    shard_0 = np.arange(
        0,
        10,
        dtype=np.uint16,
    )

    shard_1 = np.arange(
        100,
        108,
        dtype=np.uint16,
    )

    path_0 = directory / "train_00000.npy"
    path_1 = directory / "train_00001.npy"

    np.save(
        path_0,
        shard_0,
    )

    np.save(
        path_1,
        shard_1,
    )

    return [
        path_0,
        path_1,
    ]


def test_token_shard_dataset():
    """
    Verify:
        1. multiple shards are supported
        2. trailing incomplete tokens are ignored
        3. global index maps to the correct shard
        4. output dtype is torch.long
        5. output shape is [seq_len]
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        shard_paths = create_test_shards(
            tmpdir
        )

        dataset = TokenShardDataset(
            shard_paths=shard_paths,
            seq_len=4,
        )

        # shard 0 -> 2 sequences
        # shard 1 -> 2 sequences
        assert len(dataset) == 4

        sample_0 = dataset[0]["input_ids"]
        sample_1 = dataset[1]["input_ids"]
        sample_2 = dataset[2]["input_ids"]
        sample_3 = dataset[3]["input_ids"]

        assert torch.equal(
            sample_0,
            torch.tensor(
                [0, 1, 2, 3],
                dtype=torch.long,
            ),
        )

        assert torch.equal(
            sample_1,
            torch.tensor(
                [4, 5, 6, 7],
                dtype=torch.long,
            ),
        )

        assert torch.equal(
            sample_2,
            torch.tensor(
                [100, 101, 102, 103],
                dtype=torch.long,
            ),
        )

        assert torch.equal(
            sample_3,
            torch.tensor(
                [104, 105, 106, 107],
                dtype=torch.long,
            ),
        )

        assert sample_0.dtype == torch.long
        assert sample_0.shape == (4,)

        # Negative indexing should work like a normal Python sequence.
        assert torch.equal(
            dataset[-1]["input_ids"],
            sample_3,
        )


def test_dataset_summary():
    """
    Verify Dataset metadata.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        shard_paths = create_test_shards(
            tmpdir
        )

        dataset = TokenShardDataset(
            shard_paths=shard_paths,
            seq_len=4,
        )

        summary = dataset.summary()

        assert summary["num_shards"] == 2
        assert summary["seq_len"] == 4
        assert summary["total_tokens_on_disk"] == 18
        assert summary["total_sequences"] == 4

        assert summary["tokens_per_shard"] == [
            10,
            8,
        ]

        assert summary["sequences_per_shard"] == [
            2,
            2,
        ]


def test_find_token_shards():
    """
    Verify tokenized shards are discovered and sorted correctly.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        create_test_shards(
            tmpdir
        )

        # This file should not match train_*.npy.
        np.save(
            tmpdir / "val_00000.npy",
            np.arange(
                4,
                dtype=np.uint16,
            ),
        )

        paths = find_token_shards(
            directory=tmpdir,
            pattern="train_*.npy",
        )

        assert len(paths) == 2

        assert paths[0].name == "train_00000.npy"
        assert paths[1].name == "train_00001.npy"


def test_dataloader():
    """
    Verify DataLoader stacks samples into:

        input_ids: [B, T]
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        shard_paths = create_test_shards(
            tmpdir
        )

        dataset = TokenShardDataset(
            shard_paths=shard_paths,
            seq_len=4,
        )

        dataloader = build_dataloader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        batch = next(
            iter(dataloader)
        )

        input_ids = batch["input_ids"]

        assert input_ids.shape == (
            2,
            4,
        )

        assert input_ids.dtype == torch.long

        assert torch.equal(
            input_ids[0],
            torch.tensor(
                [0, 1, 2, 3],
                dtype=torch.long,
            ),
        )

        assert torch.equal(
            input_ids[1],
            torch.tensor(
                [4, 5, 6, 7],
                dtype=torch.long,
            ),
        )


if __name__ == "__main__":
    test_token_shard_dataset()
    print("test_token_shard_dataset: PASSED")

    test_dataset_summary()
    print("test_dataset_summary: PASSED")

    test_find_token_shards()
    print("test_find_token_shards: PASSED")

    test_dataloader()
    print("test_dataloader: PASSED")

    print("\nAll data tests passed.")