# LLM From Scratch (llmshin)

This project implements a decoder-only Transformer (GPT-style) from scratch using PyTorch. It is trained on the TinyShakespeare dataset and performs character-level text generation.

## Features
- **Custom Tokenizer:** Simple character-level tokenizer.
- **Transformer Architecture:** Multi-head self-attention, feed-forward layers, layer normalization, and residual connections.
- **Configurable:** Centralized hyperparameters in `config.py`.
- **Training Script:** Full training loop with periodic evaluation.
- **Inference Script:** Generate text from a prompt.

## Project Structure
- `data/`: Contains the training data (`input.txt`).
- `config.py`: Hyperparameters for the model and training.
- `tokenizer.py`: Character-level tokenization logic.
- `data_loader.py`: Loading and batching data.
- `model.py`: The Transformer model architecture.
- `train.py`: The training script.
- `sample.py`: The inference script for generating text.
- `requirements.txt`: Project dependencies.

## Getting Started

### 1. Setup Environment
It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data
The data is automatically downloaded when you run the training script, but you can also manually ensure `data/input.txt` exists.

### 3. Train the Model
Run the training script to start training the model:
```bash
python3 train.py
```
This will save `model.pth` and `vocab.json` upon completion.

### 4. Generate Text
Use the `sample.py` script to generate text from the trained model:
```bash
python3 sample.py "ROMEO: "
```

## Architecture Details
The model is a standard Transformer decoder:
- Embedding Layer (Token + Positional)
- N blocks of:
    - Layer Norm 1
    - Multi-Head Attention
    - Layer Norm 2
    - Feed Forward Network (MLP)
- Final Layer Norm
- Linear Head (Logits)
