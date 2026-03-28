import os
import math
import torch
from model_ultra import UltraGPT
from data_loader import LargeDataLoader
import config_ultra as config
from tokenizer import BPETokenizer

# Hyperparameters
batch_size = config.batch_size
block_size = config.block_size
max_iters = config.max_iters
eval_interval = config.eval_interval
learning_rate = config.learning_rate
device = config.device
eval_iters = config.eval_iters
patience_limit = config.patience if hasattr(config, 'patience') else 20
warmup_iters = config.warmup_iters if hasattr(config, 'warmup_iters') else 5000
grad_accum_steps = config.grad_accum_steps if hasattr(config, 'grad_accum_steps') else 4
min_lr = learning_rate / 10  # 10% of max LR

# Load data using LargeDataLoader
# Ensure data/train.bin and data/val.bin exist (run prepare_data.py first)
if not os.path.exists('data/train.bin'):
    print("Error: data/train.bin not found. Please run 'python prepare_data.py' first.")
    exit(1)

loader = LargeDataLoader('data', block_size, batch_size, device)

# Load tokenizer for vocab size
tokenizer = BPETokenizer.load_vocab('vocab_ultra.json')
vocab_size = tokenizer.vocab_size

# Initialize model
model = UltraGPT(vocab_size)
m = model.to(device)
print(f"Model vocab size: {vocab_size}")
print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")

# Learning rate scheduler logic


def get_lr(it):
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) If it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges from 1.0 to 0.0
    return min_lr + coeff * (learning_rate - min_lr)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')
patience_counter = 0

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch(split)
            # Use autocast for CUDA/MPS/CPU
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type == 'cuda' else torch.float32, enabled=True):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):
    # Determine the learning rate for this iteration
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:e}")

        # Early stopping check
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'model_ultra_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at step {iter} (no improvement for {patience_limit} eval intervals)")
                break

    # Gradient accumulation loop
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(grad_accum_steps):
        # sample a batch of data
        xb, yb = loader.get_batch('train')

        # evaluate the loss with mixed precision
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type == 'cuda' else torch.float32, enabled=True):
            logits, loss = model(xb, yb)
            # scale the loss to account for gradient accumulation
            loss = loss / grad_accum_steps

        loss.backward()

    optimizer.step()


# Save the model
torch.save(model.state_dict(), 'model_ultra.pth')
tokenizer.save_vocab('vocab_ultra.json')

# generate some text
print("Generating sample text...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
