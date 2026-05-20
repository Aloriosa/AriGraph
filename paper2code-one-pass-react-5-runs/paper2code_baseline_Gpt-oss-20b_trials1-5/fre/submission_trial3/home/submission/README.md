# Functional Reward Encoding (FRE) Toy Implementation

This repository contains a lightweight, fully reproducible implementation of the core ideas from
> *Unsupervised Zero‑Shot Reinforcement Learning via Functional Reward Encodings*.
The code demonstrates how to:

1. **Sample random reward functions** (goal‑reaching, linear, small MLP).
2. **Train a transformer‑based encoder** that maps a few `(state, reward)` samples to a latent
   representation `z`.
3. **Decode the reward function** from `z` for any state using a small MLP.
4. **Evaluate** the fidelity of the recovered reward function on a held‑out set of states.

> **Note**  
> This toy implementation is *not* a full reinforcement‑learning pipeline. It focuses on the
> functional‑reward‑encoding part of the paper and can be run on any machine with a CPU
> (GPU is optional). The results are illustrative only, not comparable to the numbers in the
> original paper.

## How to Run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Generate a synthetic dataset of 2‑D states.
3. Train the FRE encoder/decoder pair for a few epochs.
4. Sample a new reward function, encode it, and evaluate the decoding accuracy.
5. Save the outputs to `results/`.

All files are checked into the repository; no large artifacts are produced.

## File Overview

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `data.py` | Synthetic state dataset generation |
| `reward_prior.py` | Functions that sample random reward functions |
| `fre_encoder.py` | Encoder & decoder network definitions |
| `train_fre.py` | Training loop for FRE |
| `evaluate_fre.py` | Encoding a new reward, decoding, and evaluation |
| `reproduce.sh` | End‑to‑end reproducibility script |
| `README.md` | This document |

## Expected Output

After running `reproduce.sh` you should see something like:

```
Training FRE encoder: 100% |████████████████████| time elapsed: 00:01
Evaluation on new reward function:
  MSE on held‑out states: 0.0123
```

A `results/` directory will contain:

- `fre_model.pt` – the trained encoder/decoder state dict
- `eval_report.json` – a JSON file with the MSE score and hyperparameters

Feel free to tweak hyperparameters in `train_fre.py` or `evaluate_fre.py` to explore
different settings.