# Reproduction of “Stay on topic with Classifier‑Free Guidance”

This repository contains a minimal, self‑contained reproduction of the main idea of the paper
“Stay on topic with Classifier‑Free Guidance”.  
The goal is to show that **Classifier‑Free Guidance (CFG)** can be applied to a vanilla
autoregressive language model (GPT‑2) without any additional training, and that it
produces more prompt‑adherent generations.

## What we provide

| File | Purpose |
|------|---------|
| `reproduce.sh` | Bash script that installs requirements, downloads a model, runs a baseline generation and a CFG generation, and stores the results. |
| `cfg_demo.py` | Python script that implements CFG for a causal language model. |
| `README.md` | This file. |

## How to run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages (`torch`, `transformers`).
2. Load the `gpt2` model from Hugging‑Face.
3. Run a **baseline** generation (no CFG) and a **CFG** generation with a guidance weight `γ=1.5` on a set of short prompts.
4. Store the generated texts in `baseline.txt` and `cfg.txt`.
5. Print a short comparison summary.

All files are under 1 GB; no large artifacts are committed.

The repository is intentionally lightweight – it does **not** attempt to reproduce the full benchmark suite of the paper (which would require several large models and hours of GPU time). It demonstrates the core algorithmic idea: re‑weighting the model’s logits with the difference between a *prompt‑conditioned* and an *unconditioned* prediction.

## Expected output

After running `reproduce.sh`, you should see something like:

```
Baseline generation finished.
CFG generation finished.
Results written to baseline.txt and cfg.txt.
```

Both `baseline.txt` and `cfg.txt` contain the generated continuations for each prompt,
showing that CFG tends to keep the generation closer to the prompt.

Feel free to tweak `gamma` or add more prompts in `cfg_demo.py` to explore the effect
of CFG further.