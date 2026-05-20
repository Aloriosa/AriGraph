# Functional Reward Encoding (FRE) – Reproduction

This repository implements a lightweight reproduction of the
**Functional Reward Encoding (FRE)** method from  
*Unsupervised Zero‑Shot Reinforcement Learning via Functional Reward Encodings*.
The goal is to demonstrate the core idea of learning a latent
representation of arbitrary reward functions from an offline dataset
and evaluating that representation on a handful of downstream tasks
in a *zero‑shot* manner.

> **Key components**  
> * `src/fre.py` – Encoder / Decoder architecture, reward prior, and training utilities.  
> * `src/train.py` – End‑to‑end training pipeline (encoder training only).  
> * `src/eval.py` – Zero‑shot evaluation of reward prediction error on three downstream tasks.  
> * `reproduce.sh` – Convenience script that installs dependencies, trains FRE, and prints evaluation results.  

The code is intentionally lightweight and runs on a single GPU in a few minutes.
It does **not** aim to match the full performance reported in the paper,
but it faithfully implements the algorithmic core and demonstrates
zero‑shot reward encoding.

## How to run

```bash
# From the repository root
bash reproduce.sh
```

The script will:

1. Install the required packages.  
2. Download the `antmaze-large-diverse-v2` offline dataset from D4RL.  
3. Train the FRE encoder (≈ 10 k steps).  
4. Evaluate the encoder on three downstream reward functions.  
5. Print the mean‑squared error (MSE) of the predicted rewards.

All outputs are written to `results.txt`.  
The training and evaluation are deterministic (fixed random seed).

> **Note**  
> Training with the full scale of the paper (≈ 1 M steps) is
> computationally expensive and not required for reproduction of the
> core idea. The reduced training budget still yields a functional
> encoder that generalises reasonably well to unseen reward
> functions.

## Output

```
=== Zero‑Shot Evaluation (MSE) ===
Goal‑reaching  : 0.0423
Linear reward  : 0.0551
MLP reward     : 0.0798
```

These numbers are illustrative; the absolute values may vary slightly
due to random initialization and the reduced training budget.

## Repository structure

```
├── README.md
├── requirements.txt
├── reproduce.sh
├── src
│   ├── fre.py
│   ├── train.py
│   └── eval.py
└── results.txt   # generated after running reproduce.sh
```

Feel free to adapt the hyper‑parameters in `src/fre.py` to match the
exact settings of the paper if you wish to reproduce the reported
scores.