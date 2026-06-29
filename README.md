# GPT-2 From Scratch (PyTorch)

A clean, educational reimplementation of GPT-2 in PyTorch, built from first principles and compatible with Hugging Face pretrained checkpoints.

The goal of this project is to understand every component of a decoder-only Transformer by implementing the architecture from scratch instead of relying on high-level libraries.

## Features

* GPT-2 architecture implemented from scratch
* Multi-Head Causal Self-Attention
* Pre-LayerNorm Transformer blocks
* GPT-2 MLP with approximate GELU
* Learned token and positional embeddings
* Weight tying between input embeddings and language modeling head
* Loading pretrained GPT-2 weights from Hugging Face
* Autoregressive text generation using `torch.multinomial`
* Output verified against Hugging Face's `GPT2LMHeadModel`

## Repository Structure

```text
.
├── train_gpt2.py     # GPT-2 implementation and weight loading
├── compare.py        # Compares generation against Hugging Face GPT-2
└── README.md
```

## Model Architecture

```
Input Tokens
      │
      ▼
Token Embeddings
      +
Position Embeddings
      │
      ▼
12 × Transformer Blocks
    ├── LayerNorm
    ├── Causal Multi-Head Attention
    ├── Residual Connection
    ├── LayerNorm
    ├── MLP (Linear → GELU → Linear)
    └── Residual Connection
      │
      ▼
Final LayerNorm
      │
      ▼
Linear LM Head (Weight Tied)
      │
      ▼
Vocabulary Logits
```

## Implemented GPT-2 Configurations

|        Model | Layers | Heads | Embedding Size |
| -----------: | -----: | ----: | -------------: |
|        GPT-2 |     12 |    12 |            768 |
| GPT-2 Medium |     24 |    16 |           1024 |
|  GPT-2 Large |     36 |    20 |           1280 |
|     GPT-2 XL |     48 |    25 |           1600 |

## Installation

```bash
git clone https://github.com/DaanishSharma/GPT2
cd GPT2

pip install torch transformers tiktoken
```

## Running

Compare your implementation against Hugging Face:

```bash
python compare.py
```

Example output:

```
YOUR MODEL
Hello, I'm a language model, so I'm able to take ideas and build them...

HF MODEL
Hello, I'm a language model, so I'm able to take ideas and build them...
```

The generated outputs are identical, confirming that the implementation matches Hugging Face's GPT-2 inference behavior.

## Loading Pretrained Models

```python
from train_gpt2 import GPT

model = GPT.from_pretrained("gpt2")
```

Supported checkpoints:

* gpt2
* gpt2-medium
* gpt2-large
* gpt2-xl

## Verification

The implementation has been verified by:

* Loading official Hugging Face GPT-2 checkpoints
* Copying all pretrained weights into the custom implementation
* Matching generated text using identical sampling parameters and random seeds
* Producing the same outputs as `GPT2LMHeadModel`

## References

* Attention Is All You Need (2017)
* Improving Language Understanding by Generative Pre-Training (GPT)
* Language Models are Unsupervised Multitask Learners (GPT-2)
* Hugging Face Transformers
* Andrej Karpathy's nanoGPT
