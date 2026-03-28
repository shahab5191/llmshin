import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class BPETokenizer:
    def __init__(self, data_path=None, vocab_path=None, vocab_size=1000):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        if vocab_path and os.path.exists(vocab_path):
            self.tokenizer = Tokenizer.from_file(vocab_path)
        elif data_path and os.path.exists(data_path):
            trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
            self.tokenizer.train(files=[data_path], trainer=trainer)
        
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, s):
        return self.tokenizer.encode(s).ids

    def decode(self, l):
        return self.tokenizer.decode(l)

    def save_vocab(self, path):
        self.tokenizer.save(path)

    @classmethod
    def load_vocab(cls, path):
        return cls(vocab_path=path)

class CharTokenizer:
    def __init__(self, data_path=None, vocab=None):
        if vocab:
            self.chars = sorted(list(vocab))
        elif data_path and os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.chars = sorted(list(set(text)))
        else:
            self.chars = []
        
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def save_vocab(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.chars, f)

    @classmethod
    def load_vocab(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return cls(vocab=vocab)

if __name__ == "__main__":
    # Test the BPE tokenizer
    tokenizer = BPETokenizer(data_path='data/input.txt', vocab_size=500)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    test_str = "Before we proceed any further, hear me speak."
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
