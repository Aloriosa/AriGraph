# RICE – Reinforcement Learning with Explanation (Reproduction)

This repository contains a minimal, fully‑reproducible implementation of the **RICE** algorithm as described in the paper *“RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation”*.  
The implementation uses the `CartPole-v1` environment from OpenAI Gym and a simplified version of the algorithm that demonstrates the key ideas:

1. **Base training** – Train a PPO agent on the default initial state distribution.  
2. **Explanation (critical states)** – Collect states that lead to large future returns and treat them as *critical* (exploration frontiers).  
3. **Mixed initial distribution** – During refinement, start episodes either from the default initial state or from one of the collected critical states.  
4. **Intrinsic exploration** – Add a Random Network Distillation (RND) bonus to encourage exploration from new initial states.  

The `reproduce.sh` script installs the necessary dependencies, trains the base agent, collects critical states, refines the agent with RICE, evaluates both agents, and writes the results to `results.json`.

> **NOTE**  
> The full paper evaluates many complex environments (MuJoCo, mining, security, etc.).  
> For the sake of reproducibility in a constrained environment, this repository focuses on a lightweight example that still showcases the main components of RICE.

## How to Run

```bash
bash reproduce.sh
```

This will:

1. Install required Python packages.  
2. Train a base PPO agent (`base.zip`).  
3. Collect 200 critical states from the base agent.  
4. Train a refined PPO agent (`refine.zip`) using the mixed initial distribution and RND.  
5. Evaluate both agents and write results to `results.json`.  

The script prints intermediate progress and final average rewards.

## Repository Structure

```
├── README.md
├── reproduce.sh
├── train_base.py
├── train_refine.py
├── collect_critical.py
├── utils.py
├── env_wrappers.py
├── requirements.txt
└── results.json   (generated after running reproduce.sh)
```

- **train_base.py** – Trains the base PPO agent.  
- **collect_critical.py** – Collects critical states from the base agent.  
- **train_refine.py** – Refines the agent using RICE.  
- **env_wrappers.py** – Contains the mixed‑initial and RND wrappers.  
- **utils.py** – Utility functions (seeding, evaluation, etc.).  
- **requirements.txt** – Python dependencies.  

All code uses deterministic seeds to ensure reproducibility.

---

## Expected Output

After running `reproduce.sh`, you should see output similar to:

```
Base training finished. Avg reward over 10 eval episodes: 195.00
Collected 200 critical states.
Refinement training finished. Avg reward over 10 eval episodes: 198.50
Results written to results.json
```

The `results.json` file will contain JSON with the following structure:

```json
{
  "base_avg_reward": 195.0,
  "refine_avg_reward": 198.5,
  "critical_states_count": 200
}
```

Feel free to adjust hyper‑parameters in the scripts (`total_timesteps`, `p`, `lambda_intrinsic`, etc.) and re‑run to observe different results. The implementation remains fully reproducible.

---

## License

MIT License. See `LICENSE` for details.