import torch
from model_ultra import UltraGPT
from tokenizer import BPETokenizer
import config_ultra as config

# Hyperparameters
device = config.device

# Load tokenizer
tokenizer = BPETokenizer.load_vocab("vocab_ultra.json")
vocab_size = tokenizer.vocab_size

# Initialize model
model = UltraGPT(vocab_size)

# Load the model weights safely
# We use weights_only=True for security and compatibility
state_dict = torch.load("model_ultra.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)

model.to(device)
model.eval()


# Generate text
def sample(prompt="", max_new_tokens=500, temperature=0.8, top_k=50):
    # Encode prompt, or start with a newline if empty
    if prompt == "":
        context = torch.tensor(
            [[0]], dtype=torch.long, device=device
        )  # Start with first token
    else:
        context = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.long, device=device
        ).unsqueeze(0)

    generated = model.generate(
        context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
    )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    import sys

    prompt = sys.argv[1] if len(sys.argv) > 1 else ""
    sample(prompt)
