import torch
import torch.nn as nn
from torch.nn import functional as F
import config_ultra as config

# Hyperparameters
n_embd = config.n_embd
n_head = config.n_head
n_kv_heads = config.n_kv_heads
n_layer = config.n_layer
dropout = config.dropout
block_size = config.block_size
device = config.device


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster and simpler than LayerNorm)"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_rope_freqs(dim, seq_len, theta=10000.0):
    """Precompute Rotary Positional Embedding frequencies"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rope(x, freqs_cis):
    """Apply RoPE to a tensor"""
    # x: (B, T, n_heads, head_dim)
    # freqs_cis: (T, head_dim // 2)
    B, T, H, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, T, H, -1, 2))
    freqs_cis = freqs_cis[:T].view(1, T, 1, -1)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)


class GroupedQueryAttention(nn.Module):
    """GQA: Reduces KV heads to save memory on M4"""

    def __init__(self):
        super().__init__()
        self.head_dim = n_embd // n_head
        self.n_heads = n_head
        self.n_kv_heads = n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(n_embd, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(n_embd, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_head * self.head_dim, n_embd, bias=False)

        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        # (B, T, C) -> (B, T, H, D)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(B, T, self.n_heads, self.head_dim)
        xk = xk.view(B, T, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        xq = apply_rope(xq, freqs_cis)
        xk = apply_rope(xk, freqs_cis)

        # Repeat KV heads for GQA
        # (B, T, n_kv_heads, D) -> (B, T, n_heads, D)
        xk = xk.repeat_interleave(self.n_rep, dim=2)
        xv = xv.repeat_interleave(self.n_rep, dim=2)

        # Attention calculation
        xq = xq.transpose(1, 2)  # (B, H, T, D)
        xk = xk.transpose(1, 2)  # (B, H, T, D)
        xv = xv.transpose(1, 2)  # (B, H, T, D)

        # Efficient Flash Attention (requires PyTorch 2.0+)
        output = F.scaled_dot_product_attention(
            xq, xk, xv, is_causal=True, dropout_p=dropout if self.training else 0.0
        )

        output = output.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.wo(output))


class SwiGLUMLP(nn.Module):
    """SwiGLU Activation Function as used in Llama-2/3"""

    def __init__(self, dim):
        super().__init__()
        hidden_dim = int(2 / 3 * 4 * dim)
        # Round hidden_dim to nearest multiple of 256 for hardware alignment
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = GroupedQueryAttention()
        self.feed_forward = SwiGLUMLP(n_embd)
        self.attention_norm = RMSNorm(n_embd)
        self.ffn_norm = RMSNorm(n_embd)

    def forward(self, x, freqs_cis):
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class UltraGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([Block() for _ in range(n_layer)])
        self.norm = RMSNorm(n_embd)
        self.output = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight Tying: tie output layer weights to token embeddings
        self.output.weight = self.token_embedding.weight

        # RoPE frequencies precomputation registered as buffer
        self.register_buffer(
            "freqs_cis", precompute_rope_freqs(n_embd // n_head, block_size * 2)
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx)

        # Use precomputed RoPE frequencies from buffer
        freqs_cis = self.freqs_cis[:T]

        for layer in self.layers:
            x = layer(x, freqs_cis)

        x = self.norm(x)
        logits = self.output(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text with temperature and top-k sampling.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # focus only on the last time step and scale by temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
