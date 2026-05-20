# RICE: Reinforcement Learning Refinement with Explanation

This repository contains a minimal but functional implementation of the RICE algorithm described in
*RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation*.
It demonstrates the core pipeline:

1. **Pre‑training** a PPO agent on a MuJoCo environment (default: Hopper‑v3).
2. **Training a mask network** (our simplified StateMask) that identifies important states.
3. **Refinement** (RICE) where the agent is re‑trained:
   * with a *mixed* initial state distribution (probability `P` of starting from a
     critical state identified by the mask network, otherwise from the default reset),
   * and with an *exploration bonus* from Random Network Distillation (RND)
     weighted by `LAMBDA`.

The final script produces a `final_reward.txt` file containing the mean episode reward after
refinement. The reproduction is fully reproducible on an Ubuntu 24.04 LTS container
with an NVIDIA GPU.

## How to run

```bash
bash reproduce.sh
```

The script will:

1. Install dependencies.
2. Train the pre‑trained PPO agent (≈ 200 k timesteps).
3. Train the mask network (≈ 50 k timesteps).
4. Perform RICE refinement (≈ 200 k timesteps).
5. Evaluate the refined agent on 10 episodes and write the mean reward to `final_reward.txt`.

## Customization

- Change the environment by editing `config.py` (e.g., `ENV_NAME = "Walker2d-v3"`).
- Adjust hyper‑parameters (`TOTAL_TIMESTEPS`, `MASK_TRAIN_TIMESTEPS`, `REFINE_TIMESTEPS`,
  `P`, `LAMBDA`, `ALPHA`) in `config.py`.
- The implementation uses CPU by default; change `device='cpu'` to `'cuda'` in `main.py`
  if a GPU is available.

## Limitations

This is a **toy implementation** focusing on the core ideas of RICE. It does not
implement all experimental baselines from the paper, nor does it match the exact
experimental protocol. Nevertheless, it demonstrates:

- Training a mask network that can identify critical states.
- Using a mixed initial distribution to avoid over‑fitting.
- Adding an intrinsic exploration bonus to escape local optima.

Feel free to extend the code for more thorough experiments or to integrate
additional baselines.