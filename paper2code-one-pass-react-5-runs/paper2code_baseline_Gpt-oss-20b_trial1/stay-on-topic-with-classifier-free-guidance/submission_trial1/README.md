# Classifier‑Free Guidance (CFG) Reproduction

This repository reproduces the core idea of the paper *“Stay on topic with Classifier‑Free Guidance”* in a minimal, end‑to‑end example.  
We demonstrate how to apply CFG to a standard causal language model (GPT‑2) without any additional training or fine‑tuning.

## What the repository contains

| File | Purpose |
|------|---------|
| `reproduce.sh` | Bash script that installs the required packages, downloads the model and tokenizer, runs the inference, and writes the outputs to the `outputs/` directory. |
| `cfg_inference.py` | Python module that implements the CFG algorithm for autoregressive language models. |
| `utils.py` | Small helper functions used by the inference script. |
| `outputs/` | Directory created by `reproduce.sh` that holds the generated texts. |
| `README.md` | This file. |

## How to run

The container used for evaluation will have an NVIDIA A10 GPU already available.  
Simply run the reproduction script:

```bash
bash reproduce.sh
```

The script will:

1. Install `torch` and `transformers` (CPU‑only version is fine, GPU will be used automatically if available).  
2. Download the `distilgpt2` model (tiny GPT‑2, 82M parameters).  
3. Generate 3 example prompts with two CFG strengths:  
   * `γ = 1.0`  → vanilla inference  
   * `γ = 1.5`  → CFG inference  
4. Write the outputs to `outputs/`.  
5. Print a short summary of the results.

**Expected outcome**

After the script finishes, you should see an `outputs/` directory containing:

```
prompt_1_baseline.txt
prompt_1_cfg.txt
prompt_2_baseline.txt
prompt_2_cfg.txt
prompt_3_baseline.txt
prompt_3_cfg.txt
```

Each file contains the generated continuation of the corresponding prompt.

The script is intentionally lightweight (≈ 2 MB of source code) and does not contain any large artefacts, so it satisfies the 1 GB repository size limit.

## How the code works

The core of the CFG logic is in `cfg_inference.py`.  
For each generation step we compute:

```
logit = logit_uncond + γ * (logit_cond - logit_uncond)
```

where `logit_cond` is the model’s logits given the full prompt, and `logit_uncond` is the model’s logits when it receives only a beginning‑of‑sentence token.  
This simple re‑weighting encourages the model to stay on topic with respect to the prompt.

Feel free to experiment with other models, prompts, or γ values – the script is fully modular.

Happy reproducing! 🚀