import os
import torch
import numpy as np
from tokenizer import BPETokenizer
from datasets import load_dataset


class StreamingDataLoader:
    def __init__(
        self,
        dataset_name,
        dataset_subset,
        tokenizer,
        block_size,
        batch_size,
        device="cpu",
    ):
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        # Load separate streams for train and val to avoid data leakage
        # Use different splits if available, otherwise skip a large portion
        self.train_ds = load_dataset(
            dataset_name, name=dataset_subset, split="train", streaming=True
        )
        # Use a large skip for validation to ensure it's different data
        self.val_skip = 100000
        self.val_ds = load_dataset(
            dataset_name, name=dataset_subset, split="train", streaming=True
        ).skip(self.val_skip)

        self.train_iter = iter(self.train_ds)
        self.val_iter = iter(self.val_ds)

        self.train_buffer = []
        self.val_buffer = []

    def _fill_buffer(self, split):
        it = self.train_iter if split == "train" else self.val_iter
        buf = self.train_buffer if split == "train" else self.val_buffer
        ds = self.train_ds if split == "train" else self.val_ds

        needed = (self.block_size + 1) * self.batch_size

        while len(buf) < needed:
            try:
                example = next(it)
                tokens = self.tokenizer.encode(example["text"])
                buf.extend(tokens)
            except StopIteration:
                # Reset iterator efficiently using .skip() on the dataset object
                it = iter(ds)

        if split == "train":
            self.train_buffer, self.train_iter = buf, it
        else:
            self.val_buffer, self.val_iter = buf, it

    def get_batch(self, split="train"):
        self._fill_buffer(split)

        if split == "train":
            buf = self.train_buffer
            batch_tokens = buf[: (self.block_size + 1) * self.batch_size]
            self.train_buffer = buf[(self.block_size + 1) * self.batch_size :]
        else:
            buf = self.val_buffer
            batch_tokens = buf[: (self.block_size + 1) * self.batch_size]
            self.val_buffer = buf[(self.block_size + 1) * self.batch_size :]

        x = torch.zeros((self.batch_size, self.block_size), dtype=torch.long)
        y = torch.zeros((self.batch_size, self.block_size), dtype=torch.long)

        for i in range(self.batch_size):
            start = i * (self.block_size + 1)
            x[i] = torch.tensor(batch_tokens[start : start + self.block_size])
            y[i] = torch.tensor(batch_tokens[start + 1 : start + self.block_size + 1])

        return x.to(self.device), y.to(self.device)


class LargeDataLoader:
    def __init__(self, data_dir, block_size, batch_size, device="cpu"):
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        train_path = os.path.join(data_dir, "train.bin")
        val_path = os.path.join(data_dir, "val.bin")

        if os.path.exists(train_path) and os.path.exists(val_path):
            self.train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
            self.val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
        else:
            self.train_data = None
            self.val_data = None

    def get_batch(self, split):
        if self.train_data is None:
            raise FileNotFoundError("Local binary files not found.")

        data = self.train_data if split == "train" else self.val_data
        # Fix off-by-one: data[i+1:i+block_size+1] needs i <= len(data) - block_size - 1
        ix = torch.randint(len(data) - self.block_size - 1, (self.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + self.block_size + 1]).astype(np.int64)
                )
                for i in ix
            ]
        )

        if "cuda" in self.device:
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
                self.device, non_blocking=True
            )
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


def load_data(data_path="data/input.txt", train_split=0.9, vocab_size=1000):
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = BPETokenizer(data_path=data_path, vocab_size=vocab_size)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, tokenizer


def get_batch(data, block_size, batch_size, device="cpu"):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
