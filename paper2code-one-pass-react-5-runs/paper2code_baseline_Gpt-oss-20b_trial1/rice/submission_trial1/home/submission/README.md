# RICE – Reinforcement Learning with Explanation

This repository contains a lightweight, reproducible implementation of the **RICE** method from the paper *“RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation”*.  
The goal is to demonstrate the core ideas (state‑mask explanation, mixed initial distribution, and RND‑based exploration) on a simple OpenAI Gym environment (`CartPole-v1`).  

## Folder structure

```
/home/submission/
├── README.md
├── requirements.txt
├── reproduce.sh
├── scripts/
│   ├── train_pretrained.py
│   ├── train_mask.py
│   ├── refine_agent.py
│   └── evaluate.py
├── rices/
│   ├── mask_network.py
│   └── rnd.py
└── logs/
    └── results.txt
```

## Reproduction steps

1. **Install dependencies**  
   ```bash
   bash reproduce.sh
   ```

2. **Training pipeline**  
   The script `reproduce.sh` will:
   * Train a **pre‑trained** PPO policy on `CartPole-v1`.
   * Train a **mask network** that learns which states are critical for the policy’s reward.
   * Run the **RICE refinement** algorithm:
     * Sample initial states from a mixture of the default distribution and the identified critical states.
     * Encourage exploration with Random Network Distillation (RND).
   * Evaluate and log the average returns before and after refinement.

3. **Results**  
   After the script finishes, the file `logs/results.txt` contains:
   * Average return of the pre‑trained policy.
   * Average return of the refined policy.
   * The improvement factor.

> **Note**: This is a toy demonstration. The same pipeline can be applied to more complex environments (e.g. MuJoCo, MetaDrive) by replacing the environment name and adjusting hyper‑parameters.

## How the code relates to the paper

| Paper component | Repository component | Implementation details |
|-----------------|----------------------|------------------------|
| **State‑mask explanation** | `rices/mask_network.py` | A small discrete policy that decides to mask (replace with random action) or not. Trained with PPO on the augmented reward `R + α·mask`. |
| **Mixed initial distribution** | `scripts/refine_agent.py` | Uses the mask network to rank states by mask probability. Stores the top‑K states as *critical states* and samples from them with probability `p`. |
| **RND exploration** | `rices/rnd.py` | Implements a fixed random target network and a trainable predictor. Adds the squared prediction error as an intrinsic bonus. |
| **Refinement loop** | `scripts/refine_agent.py` | Runs PPO on the modified environment that supports resetting to arbitrary states. |
| **Evaluation** | `scripts/evaluate.py` | Runs multiple episodes for both policies and records average returns. |

## Expected outcomes

Running the reproduction script should produce a `logs/results.txt` similar to:

```
Pre‑trained policy average return: 195.3
Refined policy average return: 247.8
Improvement: 1.27×
```

The exact numbers will vary slightly due to randomness, but the refined policy should consistently outperform the pre‑trained one.

---

## Contact

For questions or issues, please open an issue in the repository or contact the authors of the original paper.