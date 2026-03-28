import os
import torch
import numpy as np
from tokenizer import BPETokenizer

class LargeDataLoader:
    def __init__(self, data_dir, block_size, batch_size, device='cpu'):
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        # Load the memory-mapped files
        self.train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+self.block_size+1]).astype(np.int64)) for i in ix])
        
        if 'cuda' in self.device:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
