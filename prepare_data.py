import os
import numpy as np
from datasets import load_dataset
from tokenizer import BPETokenizer
from tqdm import tqdm
import config_ultra as config

# Configuration
dataset_name = "HuggingFaceFW/fineweb-edu"
dataset_subset = "sample-10BT" # You can start with a smaller one if needed
num_proc = 8 # Number of processes for tokenization

# Initialize tokenizer
print("Loading/Training tokenizer...")
tokenizer = BPETokenizer(data_path='data/input.txt', vocab_size=config.vocab_size)
tokenizer.save_vocab('vocab_ultra.json')

# Load dataset
print(f"Loading dataset {dataset_name} ({dataset_subset})...")
ds = load_dataset(dataset_name, name=dataset_subset, split="train", streaming=True)

# We'll take a chunk for training
# 2 million examples is roughly 1-2 billion tokens depending on document length
max_examples = 2000000 
print(f"Processing up to {max_examples} examples...")

def tokenize_function(example):
    return {"ids": tokenizer.encode(example["text"])}

# Process and save to binary files
train_filename = os.path.join('data', 'train.bin')
val_filename = os.path.join('data', 'val.bin')

# We'll use uint16 since our vocab size is 1000 (< 65535)
dtype = np.uint16 

print("Tokenizing and writing to binary files...")
with open(train_filename, 'wb') as f_train, open(val_filename, 'wb') as f_val:
    count = 0
    for example in tqdm(ds.take(max_examples)):
        ids = tokenizer.encode(example["text"])
        ids = np.array(ids, dtype=dtype)
        
        # 90/10 split
        if count % 10 == 0:
            f_val.write(ids.tobytes())
        else:
            f_train.write(ids.tobytes())
        count += 1

print(f"Finished! Data saved to {train_filename} and {val_filename}")
