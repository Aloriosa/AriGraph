# Unsupervised Zero‑Shot Reinforcement Learning via Functional Reward Encodings (FRE)

This repository contains a minimal implementation of the FRE method described in the paper *Unsupervised Zero‑Shot Reinforcement Learning via Functional Reward Encodings*.  
The goal is to provide a reproducible pipeline that trains a Functional Reward Encoder on a random reward prior and a latent‑conditioned policy using an offline RL algorithm, then evaluates the zero‑shot policy on downstream tasks.

## Files

- `fre/functional_reward_encoding.py` – Encoder (transformer‑based) + Decoder (MLP) for functional reward encoding.  
- `rl/offline_rl.py` – A lightweight IQL‑style offline RL agent.  
- `train_fre_and_policy.py` – Orchestrates training of the FRE encoder and policy, then evaluates on a downstream task.  
- `reproduce.sh` – Shell script that installs dependencies, runs the training script, and outputs the mean reward.  
- `requirements.txt` – Python dependencies.  
- `README.md` – This documentation.

## Reproduction

From the repository root run:

```bash
bash reproduce.sh
```

The script will train the FRE encoder, train the policy, evaluate it on a zero‑shot downstream task, and write the mean reward to `output.txt`.

The entire pipeline can be executed on a CPU machine; a GPU will accelerate training but is not required.

The implementation follows the algorithmic details of the paper:
- Random reward priors (goal‑reaching, linear, MLP) are sampled and evaluated on states from the offline dataset.
- The encoder uses a permutation‑invariant transformer to produce a latent representation of the reward function.
- The decoder predicts rewards for new states given the latent code.
- The policy is trained with IQL updates conditioned on the latent reward code.
- Zero‑shot evaluation encodes a small set of reward samples and runs the policy without further training.

Feel free to adjust hyper‑parameters, dataset, or environment to explore different settings.