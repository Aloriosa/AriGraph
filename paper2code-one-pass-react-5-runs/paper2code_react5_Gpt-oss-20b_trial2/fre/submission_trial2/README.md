# Unsupervised Zero‑Shot Reinforcement Learning via Functional Reward Encodings (FRE)

This repository contains a lightweight, reproducible implementation of the *Functional Reward Encoding* (FRE) method described in:

> Kevin Frans, Seohong Park, Pieter Abbeel, Sergey Levine.  
> **Unsupervised Zero‑Shot Reinforcement Learning via Functional Reward Encodings**.  
> ICLR 2024.

The code reproduces the main experimental pipeline from the paper:

1. **FRE Encoder/Decoder** – a transformer‑based information‑bottleneck auto‑encoder that maps a small set of state–reward pairs to a latent vector `z`.
2. **Random reward prior** – a mixture of goal‑reaching, linear, and MLP rewards.
3. **Offline policy training** – an Implicit Q‑Learning (IQL) style algorithm that learns a policy conditioned on `z`.
4. **Zero‑shot evaluation** – samples a new reward function, encodes it, runs the trained policy in a new environment, and reports the return.

The repository is fully self‑contained and only depends on public Python packages. No large artefacts are committed.

---

## Repository layout

```
/home/submission/
├─ dataset_loader.py
├─ fre.py
├─ train_fre.py
├─ train_policy.py
├─ evaluate_policy.py
├─ reproduce.sh
├─ requirements.txt
├─ README.md
```

* `dataset_loader.py` – helper to load a D4RL offline dataset.
* `fre.py` – encoder/decoder modules.
* `train_fre.py` – trains the FRE encoder and decoder.
* `train_policy.py` – trains a latent‑conditioned policy with IQL.
* `evaluate_policy.py` – zero‑shot evaluation of the policy on a sampled reward.
* `reproduce.sh` – convenience script to reproduce the entire pipeline.
* `requirements.txt` – Python dependencies.

---

## How to reproduce

```bash
# 1) Install dependencies
bash reproduce.sh
```

The script will:

1. Train the FRE encoder/decoder on `halfcheetah-medium-expert-v2` for 150 k steps.
2. Train the IQL policy conditioned on the frozen encoder for 850 k steps.
3. Perform a zero‑shot evaluation on a freshly sampled reward function.

The result is written to `policy_evaluation.json`.

---

## Notes

* The training hyper‑parameters are taken directly from the paper’s Table 3.
* The offline dataset used is `halfcheetah-medium-expert-v2` (a D4RL dataset) in order to keep the example lightweight; the paper uses the ExORL dataset but the same pipeline applies.
* The code is written to run on a single NVIDIA GPU (e.g., A10) but will also run on CPU.
* The experiment is deterministic given a fixed random seed (default 42).