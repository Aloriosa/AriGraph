# Functional Reward Encoding (FRE) – Simplified Reproduction

This repository contains a lightweight, fully‑self‑contained implementation of the
core ideas from the paper *“Unsupervised Zero‑Shot Reinforcement Learning via
Functional Reward Encodings”*.
The code demonstrates how to:

1. **Generate a synthetic offline dataset** (states, actions, next‑states).
2. **Train a Functional Reward Encoder** that learns a latent representation of
   arbitrary reward functions using a small set of state‑reward samples.
3. **Evaluate the encoder** by encoding a new reward function with only 32
   samples and predicting its reward on a large test set.
4. **Report the mean‑squared‑error (MSE)** of the reward predictions as a proxy
   for the quality of the learned encoding.

> **Important**: This is *not* a full implementation of the paper’s
> offline‑RL pipeline (e.g. IQL, policy learning, AntMaze or D4RL datasets).
> The goal is to provide a runnable, reproducible reference that can be
> extended to the full method.

---

## Repository Layout

```
src/
    dataset.py   – Generates synthetic offline data.
    fre.py       – Encoder / Decoder implementation.
    trainer.py   – Trains the FRE model.
    eval.py      – Encodes a new reward and evaluates prediction error.
reproduce.sh     – Driver script that installs dependencies, trains, and evaluates.
requirements.txt – Python package requirements.
experiments/
    results.txt – Captures the final MSE.
README.md
```

---

## How to Run

```bash
# From the repository root
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Train the FRE model for a short period (≈ 1 min on CPU; GPU optional).
3. Evaluate the model on a test reward and write the MSE to
   `experiments/results.txt`.

The final output will look like:

```
=== Evaluation Results ===
Test Reward: Linear (w = [0.5, -0.3])
MSE on 10 000 test states: 0.0123
```

Feel free to modify hyper‑parameters in `src/trainer.py` or `src/eval.py`
to experiment with different settings.

---

## Extending the Implementation

* Replace the synthetic dataset with real offline datasets
  (e.g., D4RL) by modifying `dataset.py`.
* Add a policy network and train it with an off‑policy algorithm
  (e.g., IQL) using the learned latent `z` as a conditioning input.
* Use the full `fre.py` and `trainer.py` logic from the paper
  (vector discretization, reward embedding lookup, etc.).

---

## Code Overview

- **dataset.py**: Builds a small random trajectory buffer.
- **fre.py**: Implements a simple transformer‑based encoder that
  maps a set of `(state, reward)` pairs to a latent vector `z`
  and a feed‑forward decoder that predicts rewards for new states.
- **trainer.py**: Trains the encoder/decoder by minimizing the MSE
  between predicted and true rewards on a held‑out decoding set.
- **eval.py**: Samples a new reward function, encodes it from 32
  context pairs, then evaluates the decoder on 10 000 test states.

All code uses PyTorch and works on CPU or GPU (if available).

---

## License

MIT License. See the `LICENSE` file for details.