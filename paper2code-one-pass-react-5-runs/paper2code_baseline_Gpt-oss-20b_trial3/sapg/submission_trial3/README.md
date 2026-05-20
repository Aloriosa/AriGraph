# SAPG (Split & Aggregate Policy Gradients) – Minimal Reproduction

This repository contains a lightweight, fully runnable implementation of the
*Split and Aggregate Policy Gradients* (SAPG) algorithm as described in the
paper *SAPG: Split and Aggregate Policy Gradients*.
The code is intentionally kept small and self‑contained so that it can be
executed in the provided evaluation environment (Ubuntu 24.04 LTS with an
NVIDIA A10 GPU).

## Project Structure

```
/home/submission/
├── README.md
├── reproduce.sh
├── train_sapg.py
├── env_utils.py
├── policy.py
└── utils.py
```

| File | Purpose |
|------|---------|
| `reproduce.sh` | Installs dependencies and runs the training script. |
| `train_sapg.py` | Main training loop for SAPG on a vectorised CartPole environment. |
| `env_utils.py` | Helper for creating a vectorised Gym environment. |
| `policy.py` | Implements the shared backbone, latent conditioning and policy/critic heads. |
| `utils.py` | Utility functions (GAE, clipping, etc.). |

## How to Run

From the repository root:

```bash
bash reproduce.sh
```

The script will:

1. Install system packages and Python dependencies.
2. Run `train_sapg.py` with default hyper‑parameters.
3. Save a `results.json` file containing the mean episode reward after training.

The script prints progress to the console and writes a final log file
`training.log`.

> **Note**: The implementation is a lightweight demo that runs on CPU
> (GPU support is available via PyTorch if CUDA is present). The
> environment is the classic `CartPole-v1`, which is much simpler than the
> high‑dimensional manipulation tasks in the paper. The goal is to
> demonstrate the SAPG algorithmic idea – splitting a batch into
> sub‑policies, aggregating data through importance sampling, and
> maintaining a leader policy that learns from all data.

## Expected Output

After ~5 minutes the script will output:

```
Training finished.
Final mean episode reward: 195.32
Results written to results.json
```

`results.json` contains:

```json
{
  "final_mean_reward": 195.32,
  "iterations": 2000,
  "env_name": "CartPole-v1",
  "num_policies": 4
}
```

These numbers are *not* meant to match the figures in the paper – they are
just a sanity check that the algorithm runs and produces reasonable
performance on a toy problem.

## Customization

- **Number of policies (`--num_policies`)** – splits the parallel environments
  into that many blocks. The default is 4.
- **Leader‑Follower flag (`--leader_follower`)** – if set, only the first
  policy is the leader and uses data from all others. If unset, all policies
  are symmetric and use data from every other policy.
- **Entropy coefficient (`--entropy_coef`)** – encourages exploration in
  follower policies.
- **Off‑policy ratio (`--off_ratio`)** – fraction of off‑policy data to use
  in the leader’s update (default 1.0, i.e. equal amounts of on‑ and
  off‑policy data).

All hyper‑parameters can be tuned via the command line. See
`train_sapg.py --help` for details.

---

**Happy training!**