# Reproduction of *Fine‑tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem*

This repository contains a lightweight, self‑contained implementation that demonstrates the core phenomenon described in the paper: **forgetting of pre‑trained capabilities** when fine‑tuning a reinforcement learning policy on a new task.

## Experimental setup

We use a deterministic toy environment called **AppleRetrieval** that mirrors the two‑phase structure used in the paper’s Appendix A.2:

| Phase | Description |
|-------|-------------|
| **Phase 2** | A simple 1‑D grid world where the agent must move right to reach an apple (position 10). |
| **Phase 1** | The agent must first move right to a door (position 5) and then move right again to reach the apple (position 10). |

A policy is first pre‑trained on Phase 2 and then fine‑tuned on Phase 1 using five different knowledge‑retention strategies:

| Method | Description |
|--------|-------------|
| **vanilla** | Standard PPO with the pre‑trained weights as initialization. |
| **BC** | PPO + auxiliary KL‑divergence loss that keeps the policy close to the pre‑trained one on a buffer of Phase 2 states. |
| **EWC** | PPO + Elastic Weight Consolidation penalty that penalises deviation from the pre‑trained parameters weighted by the diagonal Fisher information estimated on Phase 2 data. |
| **KS** | PPO + auxiliary KL‑divergence loss computed on the online states produced by the current policy (kick‑starting). |
| **EM** | PPO + replay of Phase 2 trajectories (episodic memory). |

All experiments are performed with a small feed‑forward actor‑critic network and a simple PPO implementation.  The code is fully deterministic and runs on CPU or GPU.

## How to reproduce

```bash
bash reproduce.sh
```

The script will:

1. Install dependencies (`torch`, `numpy`).
2. Pre‑train the policy on Phase 2.
3. Fine‑tune on Phase 1 with each of the five methods.
4. Save a JSON file (`results_<method>.json`) containing the final success rate on Phase 1.

The final success rates can be inspected with:

```bash
cat results_*.json
```

## Repository structure

```
src/
│   env.py          – AppleRetrieval environment
│   policy.py       – Actor‑Critic network
│   utils.py        – Fisher computation
│   train.py        – Training script (pre‑train + fine‑tune)
requirements.txt
reproduce.sh
README.md
```

## Notes

* The implementation focuses on the toy setting to keep the code lightweight and the run time short (≈ 5 min on a CPU).
* The PPO implementation is intentionally simple; it uses clipped surrogate loss, GAE, and a value head.
* The auxiliary losses (BC, KS, EWC, EM) are implemented exactly as described in the paper’s Section C.
* The code is deterministic: setting `PYTHONHASHSEED`, `torch.backends.cudnn.deterministic`, and seeding all RNGs.
* The script outputs the final success rate for each method, which can be compared to the qualitative results reported in the paper.

Feel free to modify hyper‑parameters or extend the environment to more realistic tasks (e.g. NetHack) – the same framework can be reused with minor changes.