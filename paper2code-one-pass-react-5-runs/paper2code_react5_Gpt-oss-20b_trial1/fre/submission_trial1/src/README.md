# Functional Reward Encoding (FRE) – Mini Reproduction

This repository contains a lightweight, self‑contained implementation of the **Functional Reward Encoding (FRE)** framework described in  
*Unsupervised Zero‑Shot Reinforcement Learning via Functional Reward Encodings* (Frans et al., 2024).

The goal of this reproduction is to demonstrate the core ideas on a toy offline dataset (`CartPole-v1`) using a small random reward prior.  
It follows the high‑level algorithm from the paper:

1. **Encode** a set of `(state, reward)` samples from a random reward function with a transformer‑based VAE encoder.  
2. **Decode** rewards for unseen states with an MLP decoder.  
3. **Condition** a deterministic policy on the latent embedding `z` and train it with a simple offline RL objective (SAC‑style).  
4. **Evaluate** on new downstream reward functions by encoding a handful of samples and executing the frozen policy without further training.

> **Important** – This is a *toy* reproduction.  
> The paper trains on large, real offline datasets (e.g. ExORL, AntMaze) and uses a full‑blown Implicit Q‑Learning (IQL) algorithm.  
> Here we use a small synthetic dataset, a simple encoder‑decoder training schedule, and a SAC‑style offline RL loop for clarity and reproducibility.

## Reproduction script

```bash
bash reproduce.sh
```

The script will:

1. Install system and Python dependencies.  
2. Generate a synthetic CartPole offline dataset of 10 000 trajectories (10 k transitions).  
3. Train the FRE encoder/decoder (10 epochs).  
4. Train the policy and Q‑network (20 epochs).  
5. Save checkpoints for `encoder.pt`, `decoder.pt`, `policy.pt`, `q.pt`.  
6. Evaluate on 10 random downstream reward functions and write per‑task returns to `metrics.csv`.

The entire pipeline runs in a few minutes on a CPU‑only machine.  
If you have a GPU available, the script will automatically use it.

## Repository structure

```
├── reproducibility.sh          # driver script
├── requirements.txt            # Python dependencies
├── README.md                   # this file
├── metrics.csv                 # evaluation results
├── checkpoints/
│   ├── encoder.pt
│   ├── decoder.pt
│   ├── policy.pt
│   └── q.pt
├── data/
│   └── cartpole_offline.pkl   # synthetic offline dataset
└── src/
    ├── __init__.py
    ├── dataset_loader.py
    ├── fre.py
    ├── policy.py
    ├── reward_funcs.py
    └── train.py
```

## Expected outcome

After running `reproduce.sh` you should see a `metrics.csv` file containing the average return for each of the 10 downstream tasks and an overall mean.  
The exact numbers will vary slightly due to randomness, but the script is deterministic given the fixed seed `SEED = 42`.

Feel free to experiment with different reward priors, encoder architectures, or RL algorithms.  This code is intentionally modular to facilitate such extensions.

Enjoy experimenting with FRE!