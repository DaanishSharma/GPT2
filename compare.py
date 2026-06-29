"""
compare.py
──────────
Loads weights into your GPT (from train_gpt2.py) and HuggingFace GPT2LMHeadModel,
runs both with the same seed, and prints side-by-side text + a token-level diff.

Encoding/decoding: tiktoken gpt2 encoder (r50k_base, vocab=50257)
Sampling:          torch.multinomial — no HF pipeline anywhere
"""

import torch
import tiktoken
from transformers import GPT2LMHeadModel

from train_gpt2 import GPT, GptConfig

# ── Knobs ────────────────────────────────────────────────────────────────────
PROMPT              = "Hello, I'm a language model,"
MODEL_TYPE          = "gpt2"
MAX_NEW_TOKENS      = 30
NUM_SEQUENCES       = 5
TEMPERATURE         = 1.0   # divide logits before softmax
TOP_K               = 50    # 0 = off; keeps only top-k probability mass
SEED                = 42
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# ── Tiktoken ─────────────────────────────────────────────────────────────────
enc = tiktoken.get_encoding("gpt2")   # r50k_base — same BPE GPT-2 was trained on

def encode(text: str) -> torch.Tensor:
    """str  →  (1, T)  int64 tensor"""
    return torch.tensor(enc.encode(text), dtype=torch.long).unsqueeze(0)

def decode(token_ids) -> str:
    """(T,) list / tensor  →  str"""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return enc.decode(token_ids)

# ── Sampling loop (shared by both models) ────────────────────────────────────
@torch.no_grad()
def sample(model, idx: torch.Tensor, max_new_tokens: int,
           temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
    """
    Autoregressive generation with torch.multinomial.
    idx : (B, T)  —  prompt token ids
    Returns (B, T + max_new_tokens)
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -1024:]                           # crop to block_size

        logits = model(idx_cond)                            # (B, T, vocab)
        logits = logits[:, -1, :] / temperature             # last position only

        if top_k > 0:
            # zero out everything below the k-th largest value
            topk_vals, _ = torch.topk(logits, top_k, dim=-1)
            cutoff        = topk_vals[:, -1].unsqueeze(-1)  # (B, 1)
            logits        = logits.masked_fill(logits < cutoff, float("-inf"))

        probs      = torch.softmax(logits, dim=-1)          # (B, vocab)
        next_token = torch.multinomial(probs, num_samples=1)# (B, 1)
        idx        = torch.cat([idx, next_token], dim=1)    # (B, T+1)
    return idx

# ── HF wrapper — raw logits, no pipeline ─────────────────────────────────────
class HFWrapper(torch.nn.Module):
    """Makes HF GPT2LMHeadModel's output signature match ours (returns logits)."""
    def __init__(self, hf_model): super().__init__(); self.model = hf_model
    def forward(self, idx):       return self.model(idx).logits

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Device : {DEVICE}")
    print(f"Encoder: tiktoken gpt2  (vocab={enc.n_vocab})")

    # encode prompt and expand to B sequences
    prompt_ids = encode(PROMPT).to(DEVICE)                  # (1, T_p)
    prompt_ids = prompt_ids.expand(NUM_SEQUENCES, -1).clone()# (B, T_p)
    T_p = prompt_ids.shape[1]

    print(f"\nPrompt : \"{PROMPT}\"")
    print(f"Tokens : {encode(PROMPT).squeeze().tolist()}")
    print(f"Params : temp={TEMPERATURE}, top_k={TOP_K}, "
          f"max_new={MAX_NEW_TOKENS}, seed={SEED}, seqs={NUM_SEQUENCES}")
    sep = "─" * 64

    # ── YOUR model ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("YOUR MODEL  (train_gpt2.GPT.from_pretrained)")
    print(sep)
    your_model = GPT.from_pretrained(MODEL_TYPE).to(DEVICE)
    torch.manual_seed(SEED)
    out_yours = sample(your_model, prompt_ids.clone(),
                       MAX_NEW_TOKENS, TEMPERATURE, TOP_K)
    for i in range(NUM_SEQUENCES):
        new_tokens = out_yours[i, T_p:].tolist()
        print(f"[{i+1}] {PROMPT}{decode(new_tokens)}")

    # ── HF model ─────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("HF MODEL    (GPT2LMHeadModel, raw logits, no pipeline)")
    print(sep)
    hf_model = HFWrapper(
        GPT2LMHeadModel.from_pretrained(MODEL_TYPE)
    ).to(DEVICE)
    torch.manual_seed(SEED)
    out_hf = sample(hf_model, prompt_ids.clone(),
                    MAX_NEW_TOKENS, TEMPERATURE, TOP_K)
    for i in range(NUM_SEQUENCES):
        new_tokens = out_hf[i, T_p:].tolist()
        print(f"[{i+1}] {PROMPT}{decode(new_tokens)}")


if __name__ == "__main__":
    main()