import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BPETokenizer:
    def __init__(self, data_path=None, vocab_path=None, vocab_size=8192):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()

        if vocab_path and os.path.exists(vocab_path):
            self.tokenizer = Tokenizer.from_file(vocab_path)
        elif data_path and os.path.exists(data_path):
            # Fallback for local file
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
            )
            self.tokenizer.train(files=[data_path], trainer=trainer)

        self.vocab_size = self.tokenizer.get_vocab_size()

    @classmethod
    def train_from_iterator(cls, iterator, vocab_size=8192):
        """Train the tokenizer on a stream of text data."""
        instance = cls()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
            initial_alphabet=[],  # Start from scratch
        )

        # Train on the provided iterator
        instance.tokenizer.train_from_iterator(iterator, trainer=trainer)
        instance.vocab_size = instance.tokenizer.get_vocab_size()
        return instance

    def encode(self, s):
        return self.tokenizer.encode(s).ids

    def decode(self, l):
        return self.tokenizer.decode(l)

    def save_vocab(self, path):
        self.tokenizer.save(path)

    @classmethod
    def load_vocab(cls, path):
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(path)
        instance.vocab_size = instance.tokenizer.get_vocab_size()
        return instance
