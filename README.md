# nanoLLaDA

A **from-scratch PyTorch** “nano” implementation of **LLaDA** (*Large Language Diffusion with mAsking*) inspired by **Nie et al. (2025), “Large Language Diffusion Models”**.

This repo is intentionally minimal and educational: everything lives in a single, heavily-commented Jupyter notebook so you can read the whole pipeline end-to-end (data → masking diffusion → transformer → training → diffusion-style sampling).

---

## What’s inside

- **`nanoLLaDA.ipynb`** — the full implementation (model + training + sampling) and an explanation-driven walkthrough.
- **`LICENSE`** — MIT License.
- **`README.md`** — this document.

---

## Conceptual overview (LLaDA in one paragraph)

Instead of generating text left-to-right like a standard causal LM (GPT-style), LLaDA treats generation as a **discrete diffusion / denoising** process. The “noise” is a special **`[MASK]` token**. During training, a random masking rate \(t \sim U(0,1)\) is sampled and tokens are masked with probability \(t\). The model is trained to reconstruct the original tokens **only at masked positions**. During inference, you start from a highly-masked sequence and iteratively refine predictions while re-masking low-confidence tokens (“mask-predict”-style).

---

## Key features implemented in the notebook

### Tokenization (GPT-2 BPE + custom `[MASK]`)
- Uses **`tiktoken`** with GPT-2 encoding.
- Adds a **custom `[MASK]` token id** at `enc.n_vocab`, increasing vocab size by 1.
- Implements safe decoding by filtering mask tokens before calling `tiktoken.decode()`.

### Architecture (Transformer, but bidirectional)
The model is a small transformer with modern components:

- **Bidirectional self-attention** (important!)
  - Uses PyTorch **scaled dot-product attention** with `is_causal=False`.
  - This is a core difference vs GPT-like models.

- **RMSNorm**
- **RoPE** (Rotary Positional Embeddings)
- **SwiGLU MLP**
- **Pre-norm residual blocks**
- **Weight tying** between token embeddings and output projection

### Training objective (masked loss only)
- Input: a sequence where some tokens have been replaced with `[MASK]`.
- Target: the original clean sequence.
- Loss: **cross-entropy computed only on masked positions**.

### Diffusion utilities
- **Forward diffusion**: random masking with ratio \(t\).
- **Reverse diffusion / sampling**:
  - iterative refinement for a fixed number of steps
  - optional **temperature** and **top-k** sampling
  - **low-confidence re-masking** strategy: keep confident tokens, re-mask uncertain ones for later refinement

### Training loop (practical details)
- AdamW optimizer
- periodic evaluation via `estimate_loss()`
- optional Weights & Biases integration (`use_wandb` flag)
- checkpoint saving hooks (as configured in the notebook)

---

## Requirements

The notebook imports (at least):

- `torch`
- `numpy`
- `requests`
- `tqdm`
- `rich`
- `matplotlib`
- `tiktoken`
- `wandb` (optional; can be disabled)

---

## Quickstart

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy requests tqdm rich matplotlib tiktoken wandb
```

> Note: `wandb` is optional. The notebook has `use_wandb = False` by default.

### 2) Run the notebook

Open and run:

- `nanoLLaDA.ipynb`

Typical workflow in the notebook:
1. Download Tiny Shakespeare
2. Tokenize with GPT-2 BPE + add `[MASK]`
3. Build the LLaDA-style transformer
4. Train with random masking diffusion
5. Sample with iterative re-masking generation

---

## How generation works here (high-level)

The provided `generate(...)` function implements a reverse process:

1. Predict logits for all positions.
2. Sample candidate tokens (temperature/top-k supported).
3. Fill unknown/masked positions.
4. Compute per-token confidence and re-mask a fraction of the lowest-confidence predictions.
5. Repeat for `steps`.

This produces “global” refinement (not left-to-right), which is the key intuition behind diffusion-style language modeling.

---

## Notes / limitations

- This is a **toy / educational implementation** aimed at clarity over speed.
- The project is notebook-centric (no packaged library, no CLI).
- Dataset is **Tiny Shakespeare** (small-scale sanity checks; not meant for SOTA quality).
- Hyperparameters and model size are intentionally “nano” and adjustable in-notebook.

---

## License

MIT — see `LICENSE`.

---

## Citation / reference

If you use ideas from this repo, cite the original
