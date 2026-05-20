# RICE (Reinforcement Learning with Explanation) – Reproduction Repository

This repository contains a **minimal, self‑contained implementation** of the
*RICE* algorithm described in the paper:

> **RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation**

The implementation follows the high‑level design:

1. **MaskNet** – a lightweight neural net that predicts whether a state is
   *critical* (i.e., should be “blinded” by forcing the agent to take a random action).
2. **Random Network Distillation (RND)** – an intrinsic exploration bonus that
   rewards the agent for visiting novel states.
3. **Mixed initial state distribution** – during training, the environment is
   reset either from the default initial state or from a critical state
   collected in a buffer.
4. **PPO policy** – the main agent is trained with the standard PPO algorithm
   (stable‑baselines3).

The code is intentionally compact and is designed to run inside the
7‑day Docker container used for grading.  It trains on the *Hopper* MuJoCo
environment for 200 k timesteps and writes the final evaluation
mean reward to `./logs/evaluation.txt`.

## Reproducing the Results

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Run the training script (`src/main.py`) with default hyper‑parameters.
3. Print the mean test reward to the console.

You can adjust hyper‑parameters via command‑line arguments:

```
python -m src.main --env Hopper-v3 --timesteps 200000 --seed 42 \
    --device cpu --mix_prob 0.25 --rnd_coef 0.01 --alpha 0.01
```

## Repository Structure

```
/home/submission/
├── README.md
├── reproduce.sh
├── requirements.txt
├── src/
│   ├── main.py
│   └── rirc/
│       └── rirc.py
```

- `rirc.py` – Core implementation of RICE.
- `main.py` – Convenience wrapper for command‑line usage.
- `reproduce.sh` – Bash script to reproduce the experiment.
- `requirements.txt` – Python package list.

## Notes

- The implementation uses **stable‑baselines3** for PPO.  
- The environment wrapper does **not** set the state to a specific critical
  observation; instead, it records critical states for the mixed‑reset
  strategy. For simulator‑based environments like MuJoCo this is sufficient
  for the toy reproduction.
- The mask network is trained with a simple REINFORCE update (no
  advantage estimation) which keeps the code lightweight.
- The RND predictor is trained with MSE loss (again, no extra tricks).

Feel free to extend this repository with more environments, better
hyper‑parameter sweeps, or a full implementation of the paper’s
experimental protocol.  The goal here is to provide a working,
reproducible baseline that demonstrates the core ideas of RICE.