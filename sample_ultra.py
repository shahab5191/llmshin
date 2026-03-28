import torch
from model_ultra import UltraGPT
from tokenizer import BPETokenizer
import config_ultra as config

# Hyperparameters
device = config.device

# Load tokenizer
tokenizer = BPETokenizer.load_vocab('vocab_ultra.json')
vocab_size = tokenizer.vocab_size

# Initialize model
model = UltraGPT(vocab_size)
model.load_state_dict(torch.load('model_ultra.pth', map_location=device))
model.to(device)
model.eval()

# Generate text
def sample(prompt="", max_new_tokens=500):
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    print(tokenizer.decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()))

if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else ""
    sample(prompt)
