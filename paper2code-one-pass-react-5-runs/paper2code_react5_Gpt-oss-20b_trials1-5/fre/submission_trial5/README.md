# Functional Reward Encoding (FRE) – Unsupervised Zero‑Shot RL

This repository contains a lightweight implementation of the **Functional Reward Encoding (FRE)** method described in the paper *Unsupervised Zero‑Shot Reinforcement Learning via Functional Reward Encodings*.  
The goal is to reproduce the main ideas (reward encoding, policy conditioning, and zero‑shot evaluation) rather than to match the exact numbers from the paper.

## Project Structure

```
/
├── requirements.txt          # Python dependencies
├── reproduce.sh              # One‑liner reproduction script
├── README.md
├── fre/
│   ├── __init__.py
│   ├── model.py              # Encoder / Decoder
│   ├── reward_prior.py       # Random reward functions
│   └── dataset.py            # Offline dataset loader
├── policy.py                 # Policy & Q‑net definitions
├── train_fre.py              # Train the FRE encoder/decoder
├── train_policy.py           # Train a FRE‑conditioned policy (IQL)
└── evaluate.py               # Zero‑shot evaluation on a downstream task
```

## How to run

```bash
# 1. Install dependencies
bash reproduce.sh
```

`reproduce.sh` will

1. Train the FRE encoder/decoder on the `antmaze-large-diverse-v2` dataset (10 epochs, 500 steps per epoch).
2. Train a simple IQL policy conditioned on the learned FRE latent (5 epochs, 500 steps per epoch).
3. Run a zero‑shot evaluation on the same AntMaze environment and print the average return.

All checkpoints and logs are stored in the `checkpoints/` and `policy_checkpoints/` directories.

> **Note**: The implementation is intentionally lightweight and may not achieve the exact performance reported in the paper. It is meant as a pedagogical reference that demonstrates the core algorithmic components.

## Customization

- To try a different offline dataset or downstream task, modify the `--dataset` / `--env_id` arguments in `reproduce.sh` or the corresponding scripts.
- The reward prior can be tuned by editing `fre/reward_prior.py`.  
- Hyper‑parameters (learning rates, batch sizes, etc.) are exposed in the training scripts’ argument parsers.

## Reproducibility

- Random seeds are fixed (`torch.manual_seed(0)` and `np.random.seed(0)`), but the training schedule is still stochastic due to GPU parallelism and the underlying dataset sampling.
- The code is fully deterministic on CPU; on GPU results may vary slightly across runs.