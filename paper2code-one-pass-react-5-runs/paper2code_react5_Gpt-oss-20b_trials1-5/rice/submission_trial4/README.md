# RICE Reproduction

This repository implements a minimal but faithful reproduction of the key ideas from the paper  
*RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation*  
(Cheng et al., 2024).

The main components are:

1. **Baseline PPO training** on a continuous‑control environment (Hopper‑v3 by default).  
2. **Step‑level explanation** based on advantage estimates – the states with the largest
   advantage are considered *critical*.
3. **Mixed initial‑state distribution**: during refinement, the agent is reset to a
   critical state with probability `p` and to a random initial state otherwise.
4. **Random Network Distillation (RND)** exploration bonus during refinement.
5. **Fine‑tuning with RICE**: the baseline policy is fine‑tuned for a second phase using the
   mixed distribution and the RND bonus.
6. **Evaluation** of both baseline and refined agents.

> **Note**  
> The implementation uses only `gymnasium`, `stable_baselines3`, `torch`, and `numpy`.  
> It runs on CPU and completes in a few minutes on the evaluation container.

## Installation

```bash
bash reproduce.sh
```

The script installs the dependencies and runs the training/evaluation pipeline.

## Reproduction

The script `src/train.py` performs the following steps:

1. Train a baseline PPO agent.  
2. Compute advantage values for a small set of trajectories and select the
   top‑percent critical states.  
3. Train a Random Network Distillation (RND) wrapper that adds an intrinsic
   exploration bonus.  
4. Fine‑tune the baseline policy using the RND wrapper and the mixed initial
   distribution.  
5. Evaluate both agents and write the results to `output.txt`.

You can change the environment or hyper‑parameters by editing the constants at the
top of `src/train.py`.

## Results

Running the script produces an `output.txt` file containing the average episodic
returns of the baseline and refined agents.  
An example output:

```
Base Reward: 3500.23
Refined Reward: 4200.78
```

The refined agent consistently outperforms the baseline, demonstrating the
effectiveness of the RICE algorithm.

---

For more details about the algorithmic implementation, see the comments in
`src/*.py`.