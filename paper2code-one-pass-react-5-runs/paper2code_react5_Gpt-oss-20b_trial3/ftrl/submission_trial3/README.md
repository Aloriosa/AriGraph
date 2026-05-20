# Reproduction of “Fine‑tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem”

This repository contains a lightweight but faithful implementation of the key ideas of the paper:

* **Toy environments** – a two‑state MDP and an AppleRetrieval grid‑world that exhibit forgetting of pre‑trained capabilities.
* **PPO agent** – a vanilla PPO implementation with optional knowledge‑retention wrappers:
  * **Elastic Weight Consolidation (EWC)**
  * **Behavioural Cloning (BC)**
  * **Kick‑starting (KS)**
  * **Episodic Memory (EM)**
* **Pre‑training** – behavioural cloning on a rule‑based policy that mimics the expert behaviour in a subset of the downstream task.
* **Fine‑tuning** – start from the pre‑trained policy and run PPO while optionally applying one of the above wrappers.
* **Evaluation** – mean return and success rate over 10 episodes.
* **Reproducibility** – `reproduce.sh` runs the whole pipeline end‑to‑end on CPU.  
  The experiments are toy‑scale but reproduce the *qualitative* behaviour described in the paper (forgetting, benefit of knowledge‑retention, etc.).

> **Note**  
> The full NetHack, Montezuma’s Revenge and Meta‑World experiments are out of scope for a CPU‑only Docker image.  
> The toy environments below capture the behavioural phenomena described in the paper and can be trained in a few minutes.

## How to run

```bash
# 1. Pre‑train on the far part of AppleRetrieval (phase 1)
bash reproduce.sh pretrain

# 2. Fine‑tune on the full AppleRetrieval with different knowledge‑retention methods
bash reproduce.sh finetune

# 3. The results are stored in `results/summary.csv` and individual logs in `results/`.
```

The script automatically installs the required packages, trains the agent and prints a short summary of the results.

## Repository layout

```
├─ README.md
├─ reproduce.sh
├─ requirements.txt
├─ envs/
│   ├─ __init__.py
│   ├─ apple_retrieval.py
│   └─ two_state_mdp.py
├─ algo/
│   ├─ __init__.py
│   ├─ ppo.py
│   ├─ ewc.py
│   ├─ bc.py
│   ├─ ks.py
│   └─ em.py
└─ train.py
```

All source code is written in pure Python with PyTorch 2.2.1 and Gymnasium 0.29.1.  
No GPU is required; the implementation is fully CPU‑compatible.

```