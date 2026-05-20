# Functional Reward Encoding (FRE) – Minimal Reproduction

This repository contains a minimal, fully‑self‑contained implementation that demonstrates the core ideas of the
*Unsupervised Zero‑Shot Reinforcement Learning via Functional Reward Encodings* paper.

## Highlights

* **FRE Encoder** – a small permutation‑invariant transformer that maps a set of
  `(state, reward)` pairs to a latent vector `z`.
* **FRE Decoder** – a lightweight MLP that predicts rewards for new states
  given `z`.
* **Training Pipeline** – randomly generates synthetic offline trajectories
  and a mixture of reward functions (goal‑reaching, linear, MLP) to train
  the encoder/decoder pair.
* **Reproduction Script** – `reproduce.sh` installs the required packages,
  runs the training and saves a CSV of the learned latent embeddings and a
  few decoded reward predictions.

> **Note**:  This is a *toy* implementation meant to be reproducible in a
> short amount of time (under 10 minutes) and to showcase the learning
> procedure.  It is **not** a full reproduction of the paper’s benchmark
> results.

## How to Run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Train the FRE encoder/decoder on synthetic data.
3. Generate a few **encoded latent vectors** for random reward functions
   and the corresponding **decoded reward predictions**.
4. Save the results to `results/fre_results.csv`.

The output file contains:

| reward_type | goal_vector (if applicable) | latent_z (32‑dim) | decoded_reward (mean) |
|-------------|-----------------------------|-------------------|-----------------------|

You can inspect the CSV with any text editor or spreadsheet tool.

## Repository Structure

```
├── README.md
├── reproduce.sh
├── requirements.txt
├── results/
│   └── fre_results.csv
├── fre/
│   ├── __init__.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── dataset.py
│   ├── train_fre.py
│   └── utils.py
```

## What was reproduced?

- The **encoder learning objective** (information bottleneck with KL penalty).
- The **decoder training** that predicts rewards for unseen states.
- The **random reward function prior** described in the paper (goal‑reaching,
  linear, MLP).
- The **overall training loop** that first trains the encoder, then freezes it
  and trains the decoder.

This simplified pipeline demonstrates the feasibility of learning a functional
reward encoding and using it to infer rewards from a small set of samples,
which is the core of FRE.

## Extending the Code

Feel free to extend the implementation to:

- Use real offline datasets (e.g. D4RL).
- Replace the toy MLP decoder with a transformer‑based decoder.
- Add a simple policy that conditioned on `z` to solve a downstream task.

The code is intentionally lightweight and modular to facilitate such
experiments.

---