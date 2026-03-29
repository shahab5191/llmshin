import os
import math
import torch
import torch.nn as nn
from model_ultra import UltraGPT
from data_loader import StreamingDataLoader
import config_ultra as config
from tokenizer import BPETokenizer
from tqdm import tqdm

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
min_lr = learning_rate / 10 

# Initialize/Train tokenizer
vocab_path = 'vocab_ultra.json'
if not os.path.exists(vocab_path):
    from datasets import load_dataset
    print("Training tokenizer on a sample of FineWeb-Edu for modern English...")
    sample_ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    def batch_iterator():
        for example in tqdm(sample_ds.take(50000), total=50000):
            yield example["text"]
    
    tokenizer = BPETokenizer.train_from_iterator(batch_iterator(), vocab_size=config.vocab_size)
    tokenizer.save_vocab(vocab_path)
    print(f"Vocab saved! Size: {tokenizer.vocab_size}")
else:
    print(f"Loading existing tokenizer from {vocab_path}...")
    tokenizer = BPETokenizer.load_vocab(vocab_path)

vocab_size = tokenizer.vocab_size

# Initialize Streaming DataLoader
loader = StreamingDataLoader(
    dataset_name="HuggingFaceFW/fineweb-edu",
    dataset_subset="sample-10BT",
    tokenizer=tokenizer,
    block_size=block_size,
    batch_size=batch_size,
    device=device
)

# Initialize model
model = UltraGPT(vocab_size)

# Better weight initialization for stability
def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

model.apply(_init_weights)
model.to(device)

# Note: Disabling torch.compile temporarily because Inductor 
# has limited support for complex RoPE operations in some environments.
# if hasattr(torch, 'compile'):
#     model = torch.compile(model)

print(f"Model vocab size: {vocab_size}")
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

# Learning rate scheduler logic
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * max(1, it) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')
patience_counter = 0
start_iter = 0

# Check for existing checkpoint to resume
checkpoint_path = 'model_ultra_best.pth'
if os.path.exists(checkpoint_path):
    print(f"Resuming from checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_iter = checkpoint['iter'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resuming from step {start_iter} (best val loss: {best_val_loss:.4f})")

print(f"Pre-fetching {eval_iters} validation batches...")
val_batches = [loader.get_batch('val') for _ in range(eval_iters)]

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    val_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = val_batches[k]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type == 'cuda' else torch.float32, enabled=True):
            logits, loss = model(X, Y)
        val_losses[k] = loss.item()
    out['val'] = val_losses.mean()
    
    train_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = loader.get_batch('train')
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type == 'cuda' else torch.float32, enabled=True):
            logits, loss = model(X, Y)
        train_losses[k] = loss.item()
    out['train'] = train_losses.mean()
    
    model.train()
    return out

print("Starting training...")
for iter in range(start_iter, max_iters):
    # Determine and set learning rate
    lr = get_lr(iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:e}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            
            # Filter config for pickling safety
            safe_config = {k: v for k, v in config.__dict__.items() 
                          if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
            
            checkpoint = {
                'model': getattr(model, '_orig_mod', model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': iter,
                'best_val_loss': best_val_loss,
                'config': safe_config
            }
            torch.save(checkpoint, 'model_ultra_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at step {iter}")
                break

    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(grad_accum_steps):
        xb, yb = loader.get_batch('train')
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type == 'cuda' else torch.float32, enabled=True):
            logits, loss = model(xb, yb)
            loss = loss / grad_accum_steps
        loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

torch.save(getattr(model, '_orig_mod', model).state_dict(), 'model_ultra.pth')
print("Generating sample text...")
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=50)
print(tokenizer.decode(generated[0].tolist()))
