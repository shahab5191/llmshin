import os
import torch
from model_ultra import UltraGPT
from tokenizer import BPETokenizer
import config_ultra as config

# Define global variables
model = None
device = None
tokenizer = None

def load_model():
    global model, device, tokenizer

    device = config.device
    print(f"Using device: {device}")

    if not os.path.exists("vocab_ultra.json"):
        raise FileNotFoundError("vocab_ultra.json not found!")

    tokenizer = BPETokenizer.load_vocab("vocab_ultra.json")
    vocab_size = tokenizer.vocab_size

    model = UltraGPT(vocab_size)

    # Try to load the best model first, then fall back to the final one
    model_path = 'model_ultra_best.pth' if os.path.exists('model_ultra_best.pth') else 'model_ultra.pth'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}!")

    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")


def request(
    prompt: str,
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 50
) -> str:
    if not tokenizer or not model:
        raise Exception("First load the model")

    if prompt.strip() == "":
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        context = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.long, device=device
        ).unsqueeze(0)

    # Note: UltraGPT.generate already has @torch.no_grad()
    generated = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )

    return tokenizer.decode(generated[0].tolist())


def chat_loop():
    print("Welcome to UltraGPT Chat! (Type 'exit' to quit)")
    print("-" * 30)

    chat_str: str = ""
    while True:
        user_input = input("User: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Append user input to history
        chat_str += f"\nUser: {user_input}\nAI:"

        # Get AI response
        ai_answer = request(user_input)

        # Strip the prompt from the answer if the model repeats it
        # (Though our current generate() implementation returns full context + new tokens)
        new_content = ai_answer[len(chat_str)-4:] # -4 to account for the prompt suffix we added

        print(f"AI: {new_content}")
        print()

        # Update chat history for next turn
        chat_str += f" {new_content}"


if __name__ == "__main__":
    try:
        load_model()
        chat_loop()
    except Exception as e:
        print(f"Error: {e}")
