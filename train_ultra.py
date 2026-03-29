import os
import math
import torch
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
patience_limit = config.patience if hasattr(config, "patience") else 20
warmup_iters = config.warmup_iters if hasattr(config, "warmup_iters") else 5000
grad_accum_steps = config.grad_accum_steps if hasattr(config, "grad_accum_steps") else 4
min_lr = learning_rate / 10  # 10% of max LR

# Initialize/Train tokenizer
vocab_path = "vocab_ultra.json"
if not os.path.exists(vocab_path):
    from datasets import load_dataset

    print("Training tokenizer on a sample of FineWeb-Edu for modern English...")
    sample_ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )

    def batch_iterator():
        for example in tqdm(sample_ds.take(50000), total=50000):
            yield example["text"]

    tokenizer = BPETokenizer.train_from_iterator(
        batch_iterator(), vocab_size=config.vocab_size
    )
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
    device=device,
)

# Initialize model
model = UltraGPT(vocab_size)
model.to(device)

# Optional: compile the model for speedup (PyTorch 2.0+)
if hasattr(torch, "compile"):
    print("Compiling model...")
    model = torch.compile(model)

print(f"Model vocab size: {vocab_size}")
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")


# Learning rate scheduler logic
def get_lr(it):
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        # Avoid 0 LR at step 0 by using max(1, it)
        return learning_rate * max(1, it) / warmup_iters
    # 2) If it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best_val_loss = float("inf")
patience_counter = 0

# Fix the validation set for consistent evaluation
print(f"Pre-fetching {eval_iters} validation batches for consistent evaluation...")
val_batches = [loader.get_batch("val") for _ in range(eval_iters)]


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    device_type = "cuda" if "cuda" in device else "cpu"

    # Evaluate on fixed validation batches
    val_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = val_batches[k]
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16 if device_type == "cuda" else torch.float32,
            enabled=True,
        ):
            logits, loss = model(X, Y)
        val_losses[k] = loss.item()
    out["val"] = val_losses.mean()

    # For train loss, we still sample live to avoid overhead
    train_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = loader.get_batch("train")
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16 if device_type == "cuda" else torch.float32,
            enabled=True,
        ):
            logits, loss = model(X, Y)
        train_losses[k] = loss.item()
    out["train"] = train_losses.mean()

    model.train()
    return out


print("Starting training...")
for iter in range(max_iters):
    # Determine and set learning rate
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Evaluation and Checkpointing
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:e}"
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            patience_counter = 0
            # Save full checkpoint for resumption
            raw_model = getattr(model, "_orig_mod", model)
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": iter,
                "best_val_loss": best_val_loss,
                "config": config.__dict__,
            }
            torch.save(checkpoint, "model_ultra_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at step {iter}")
                break

    # Training step with Gradient Accumulation
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(grad_accum_steps):
        xb, yb = loader.get_batch("train")
        device_type = "cuda" if "cuda" in device else "cpu"
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16 if device_type == "cuda" else torch.float32,
            enabled=True,
        ):
            logits, loss = model(xb, yb)
            loss = loss / grad_accum_steps
        loss.backward()

    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# Final save
raw_model = getattr(model, "_orig_mod", model)
torch.save(raw_model.state_dict(), "model_ultra.pth")
print("Generating sample text...")
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=50)
print(tokenizer.decode(generated[0].tolist()))
