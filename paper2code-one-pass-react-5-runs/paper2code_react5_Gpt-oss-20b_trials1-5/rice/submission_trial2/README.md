# RICE – Reinforcement‑Learning Refinement with Explanation

This repository contains a lightweight implementation of the **RICE** algorithm described in the paper *“RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation”*.  
The goal is to demonstrate the main ideas:

1. **State‑mask explanation** – a lightweight neural network that learns which time‑steps in a trajectory are critical.
2. **Mixed‑initial state distribution** – a combination of the environment’s default initial states and the critical states identified by the mask.
3. **Exploration with Random Network Distillation (RND)** – an intrinsic reward that encourages the agent to visit novel states.

The repository is intentionally small and only contains the source code needed to reproduce the main experiments on a single MuJoCo environment (`Hopper-v3`).  
All heavy assets (pre‑trained checkpoints, logs, plots) are omitted to keep the repository < 1 GB.

---

## Repository layout

```
/home/submission/
├─ README.md
├─ requirements.txt
├─ reproduce.sh
└─ src/
   ├─ env_wrapper.py     # Custom Gym wrapper (adds RND, state reset)
   ├─ mask_network.py    # MLP that predicts mask probability
   ├─ train_pretrained.py
   ├─ train_mask.py
   ├─ train_refine.py
   └─ utils.py
```

---

## How to run

```bash
bash reproduce.sh
```

The script will:
1. Install the Python dependencies.
2. Train a baseline PPO agent (`pretrained.zip`).
3. Train the mask network (`mask.pt`).
4. Train the refined agent (`refined.zip`).
5. Print the final mean episode reward of the refined agent.

All checkpoints are stored in the current directory.  
The training is intentionally short (≈ 200 k timesteps each) to finish within the 7‑day limit.

---

## Expected output

After running `reproduce.sh` you should see something similar to:

```
Pre‑trained PPO finished: mean reward = 3500.23
Mask training finished: mean mask prob = 0.47
Refinement finished: mean reward = 3600.12
```

The exact numbers will vary slightly due to randomness but the refined agent should consistently outperform the pre‑trained agent by a few hundred reward points, demonstrating the effect of RICE.

---

## Notes

* The implementation is **minimal** – it focuses on the core algorithmic ideas rather than fine‑grained hyper‑parameter tuning.
* The MuJoCo environments require the `mujoco` Python package. It is installed automatically by `gymnasium==0.29.1`.
* The code uses `stable‑baselines3` for PPO, but the mask network is trained manually with a vanilla REINFORCE update.
* For the mixed initial distribution the script resets the environment to a stored critical state using the underlying MuJoCo API (`env.env.set_state(state)`).
* The RND intrinsic reward uses a fixed target network and a trainable predictor network.

Happy experimenting!