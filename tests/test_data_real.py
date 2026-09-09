from myqwen.data import TokenShardDataset, build_dataloader

paths = [
    "data/fineweb_tokenized/train_00000.npy",
    "data/fineweb_tokenized/train_00001.npy",
]

dataset = TokenShardDataset(
    shard_paths=paths,
    seq_len=512,
)

print(dataset.summary())

loader = build_dataloader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

batch = next(iter(loader))

print(batch["input_ids"].shape)
print(batch["input_ids"].dtype)
print(batch["input_ids"][0, :20])