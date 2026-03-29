import torch

# Scaled Hyperparameters for M4 Performance
batch_size = 16  # Micro-batch size per forward pass
grad_accum_steps = 4  # 16 * 4 = 64 effective batch size
block_size = 512
max_iters = 200000  # Enough to see ~6.5 billion tokens (200k steps * 64 batch * 512)
eval_interval = 1000
learning_rate = 6e-4  # Higher LR for a larger effective batch size
warmup_iters = 5000
eval_iters = 200
patience = 20

# Architecture (Llama-style scaling)
n_embd = 768
n_head = 12
n_kv_heads = 4  # Grouped-Query Attention ratio (3 Queries per 1 K/V)
n_layer = 12
dropout = 0.1
vocab_size = 8192  # Optimal for 100M parameter model


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Optimization for RTX 3060 (Ampere)
if device == "cuda":
    torch.set_float32_matmul_precision("high")  # Enables Tensor Cores
