# SAPG – Split and Aggregate Policy Gradients (Reimplementation)

This repository contains a lightweight, self‑contained implementation of the **Split and Aggregate Policy Gradients (SAPG)** algorithm described in the paper *“SAPG: Split and Aggregate Policy Gradients”* (ICML 2024).  
The goal is to demonstrate the core idea – training multiple policies on disjoint blocks of parallel environments, aggregating data from all policies via importance‑weighted clipped surrogate updates, and using a shared backbone with per‑policy latent conditioning – in a single‑GPU setting.

> **Note**: The original paper reports results on tens of thousands of GPU–simulated environments (IsaacGym).  
> Here we illustrate the method on a small benchmark (`HalfCheetah-v4`) with 256 parallel environments and 4 policies.  
> This *does not* reproduce the exact numbers from the paper, but it faithfully implements the algorithmic structure and produces a working training pipeline that can be extended to the original environments and scale.

## Quick Start

```bash
# From the repository root
bash reproduce.sh
```

The script will:

1. Install the required Python dependencies.
2. Train SAPG for 10 epochs on the `HalfCheetah-v4` benchmark.
3. Log training statistics to `logs/train.log`.
4. After training, run a short evaluation and output the mean episode return.

The training is intentionally short (≈ 5 min on a single NVIDIA A10 GPU) so that it can be run in the evaluation container.

## Project Structure

```
├── README.md
├── reproduce.sh
├── requirements.txt
├── sapg.py          # Core implementation of SAPG
└── train.py         # Training and evaluation script
```

## Hyper‑parameters

| Hyper‑parameter | Value |
|-----------------|-------|
| `M` (policies) | 4 |
| `N` (envs) | 256 |
| `steps_per_update` | 16 |
| `batch_size` | 64 (per policy) |
| `gamma` | 0.99 |
| `lam` (GAE lambda) | 0.95 |
| `lr` | 5e-4 |
| `eps_clip` | 0.2 |
| `lambda_off` | 1.0 |
| `latent_dim` | 32 |
| `epochs` | 10 |

All other defaults are inherited from the paper’s hyper‑parameter tables (e.g. learning rate, clipping factor, etc.).

## Extending to the Paper Environments

To run on the original tasks (e.g. `AllegroKukaRegrasp`), replace the environment creation in `train.py` with the appropriate IsaacGym or MuJoCo wrapper.  
The SAPG logic is independent of the simulator; only the environment interface (`gymnasium.Env`) changes.

## License

This code is released under the MIT license.  
Feel free to use, modify, and extend it for your own research.