from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken


GPT2_EOT_TOKEN_ID = 50256
UINT16_MAX = np.iinfo(np.uint16).max


def infer_next_shard_index(
    output_dir: str | Path,
    prefix: str = "train",
) -> int:
    """
    Infer the next output shard index from existing .npy files.

    Examples
    --------
    Empty / nonexistent directory:
        -> 0

    Existing files:
        train_00000.npy
        train_00001.npy
        train_00007.npy

        -> 8

    Only files that exactly match:
        {prefix}_<integer>.npy

    are considered.
    """

    output_dir = Path(
        output_dir
    ).expanduser().resolve()

    if not output_dir.exists():
        return 0

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"Expected output directory: {output_dir}"
        )

    pattern = re.compile(
        rf"^{re.escape(prefix)}_(\d+)\.npy$"
    )

    indices: list[int] = []

    for path in output_dir.iterdir():
        if not path.is_file():
            continue

        match = pattern.match(
            path.name
        )

        if match is None:
            continue

        indices.append(
            int(match.group(1))
        )

    if not indices:
        return 0

    return max(indices) + 1


class TokenShardWriter:
    """
    Incrementally accumulate token ids and write fixed-size .npy shards.
    """

    def __init__(
        self,
        output_dir: str | Path,
        tokens_per_shard: int,
        prefix: str = "train",
        start_index: int | None = None,
    ):
        if tokens_per_shard <= 0:
            raise ValueError(
                "tokens_per_shard must be positive"
            )

        self.output_dir = Path(
            output_dir
        ).expanduser().resolve()

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.tokens_per_shard = (
            tokens_per_shard
        )
        self.prefix = prefix

        if start_index is None:
            start_index = infer_next_shard_index(
                output_dir=self.output_dir,
                prefix=self.prefix,
            )

        if start_index < 0:
            raise ValueError(
                "start_index must be >= 0"
            )

        self.shard_index = (
            start_index
        )

        self.buffer = np.empty(
            self.tokens_per_shard,
            dtype=np.uint16,
        )

        self.buffer_size = 0
        self.total_tokens_written = 0
        self.total_shards_written = 0

    def _shard_path(
        self,
    ) -> Path:
        return (
            self.output_dir
            / f"{self.prefix}_{self.shard_index:05d}.npy"
        )

    def _flush_full_shard(
        self,
    ) -> None:
        if (
            self.buffer_size
            != self.tokens_per_shard
        ):
            raise RuntimeError(
                "Internal error: attempted to flush "
                "a non-full shard"
            )

        path = self._shard_path()

        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing shard: {path}"
            )

        np.save(
            path,
            self.buffer,
        )

        self.total_tokens_written += (
            self.buffer_size
        )
        self.total_shards_written += 1

        self.shard_index += 1
        self.buffer_size = 0

        print(
            f"[write] {path.name} | "
            f"{self.tokens_per_shard:,} tokens"
        )

    def add_tokens(
        self,
        token_ids: list[int] | np.ndarray,
    ) -> None:
        if len(token_ids) == 0:
            return

        token_array = np.asarray(
            token_ids,
            dtype=np.int64,
        )

        if token_array.min() < 0:
            raise ValueError(
                "Token ids must be non-negative"
            )

        if token_array.max() > UINT16_MAX:
            raise ValueError(
                "Token id exceeds uint16 range: "
                f"{int(token_array.max())}"
            )

        token_array = token_array.astype(
            np.uint16,
            copy=False,
        )

        offset = 0

        while offset < len(token_array):
            remaining_space = (
                self.tokens_per_shard
                - self.buffer_size
            )

            take = min(
                remaining_space,
                len(token_array) - offset,
            )

            self.buffer[
                self.buffer_size:
                self.buffer_size + take
            ] = token_array[
                offset:
                offset + take
            ]

            self.buffer_size += take
            offset += take

            if (
                self.buffer_size
                == self.tokens_per_shard
            ):
                self._flush_full_shard()

    def finalize(
        self,
        save_remainder: bool = True,
    ) -> None:
        if self.buffer_size == 0:
            return

        if not save_remainder:
            print(
                "[skip] final partial shard | "
                f"{self.buffer_size:,} tokens"
            )
            return

        path = self._shard_path()

        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing shard: {path}"
            )

        np.save(
            path,
            self.buffer[
                :self.buffer_size
            ].copy(),
        )

        print(
            f"[write] {path.name} | "
            f"{self.buffer_size:,} tokens "
            "(final partial shard)"
        )

        self.total_tokens_written += (
            self.buffer_size
        )
        self.total_shards_written += 1

        self.shard_index += 1
        self.buffer_size = 0


def find_parquet_files(
    input_dir: str | Path,
    pattern: str = "*.parquet",
) -> list[Path]:
    input_dir = Path(
        input_dir
    ).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {input_dir}"
        )

    paths = sorted(
        input_dir.glob(pattern)
    )

    if len(paths) == 0:
        raise FileNotFoundError(
            f"No Parquet files matching '{pattern}' "
            f"found in {input_dir}"
        )

    return paths


def tokenize_parquet_files(
    parquet_paths: list[Path],
    output_dir: str | Path,
    tokens_per_shard: int,
    batch_size: int = 1024,
    prefix: str = "train",
) -> None:
    """
    Convert FineWeb-style Parquet files into continuous uint16
    token-stream .npy shards.

    Output shard numbering is inferred automatically from output_dir.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    encoding = tiktoken.get_encoding(
        "gpt2"
    )

    start_index = infer_next_shard_index(
        output_dir=output_dir,
        prefix=prefix,
    )

    print(
        f"Output shard start index: "
        f"{start_index}"
    )

    writer = TokenShardWriter(
        output_dir=output_dir,
        tokens_per_shard=tokens_per_shard,
        prefix=prefix,
        start_index=start_index,
    )

    total_documents = 0

    for file_index, parquet_path in enumerate(
        parquet_paths
    ):
        print(
            f"\n[file {file_index + 1}/{len(parquet_paths)}] "
            f"{parquet_path}"
        )

        parquet_file = pq.ParquetFile(
            parquet_path
        )

        schema_names = (
            parquet_file.schema_arrow.names
        )

        if "text" not in schema_names:
            raise KeyError(
                f"'text' column not found in {parquet_path}. "
                f"Available columns: {schema_names}"
            )

        for record_batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["text"],
        ):
            texts = (
                record_batch
                .column(0)
                .to_pylist()
            )

            valid_texts = [
                text
                for text in texts
                if isinstance(text, str)
                and len(text) > 0
            ]

            if not valid_texts:
                continue

            encoded_batch = (
                encoding.encode_ordinary_batch(
                    valid_texts
                )
            )

            for token_ids in encoded_batch:
                token_ids.append(
                    GPT2_EOT_TOKEN_ID
                )

                writer.add_tokens(
                    token_ids
                )

            total_documents += len(
                encoded_batch
            )

            if (
                total_documents % 10_000
                < len(encoded_batch)
            ):
                print(
                    f"[progress] "
                    f"{total_documents:,} documents processed"
                )

    writer.finalize(
        save_remainder=True
    )

    print("\nDone.")
    print(
        f"documents: {total_documents:,}"
    )
    print(
        f"tokens written: "
        f"{writer.total_tokens_written:,}"
    )
    print(
        f"shards written: "
        f"{writer.total_shards_written:,}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize FineWeb-Edu Parquet files into "
            "continuous uint16 NumPy token shards."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/fineweb_raw",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/fineweb_tokenized",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.parquet",
    )

    parser.add_argument(
        "--tokens-per-shard",
        type=int,
        default=50_000_000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="train",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parquet_paths = find_parquet_files(
        input_dir=args.input_dir,
        pattern=args.pattern,
    )

    print(
        f"Found {len(parquet_paths)} Parquet file(s)."
    )

    tokenize_parquet_files(
        parquet_paths=parquet_paths,
        output_dir=args.output_dir,
        tokens_per_shard=args.tokens_per_shard,
        batch_size=args.batch_size,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
