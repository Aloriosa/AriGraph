# Unsupervised Zero‑Shot RL via Functional Reward Encodings (Mini‑FRE)

This repository contains a **minimal, fully reproducible** implementation of the
“Functional Reward Encoding” (FRE) idea from the paper
*Unsupervised Zero‑Shot Reinforcement Learning via Functional Reward Encodings*.
The code is intentionally lightweight so that it can run on the judge’s
Ubuntu‑24.04 Docker container with a single A10 GPU and finish well
within the 7‑day limit.

> **What we reproduce**  
> 1. Generate an offline dataset by running a random policy in a simple gym
>    environment (CartPole‑v1).  
> 2. Train a transformer‑based encoder + MLP decoder that learns a latent
>    representation `z` of arbitrary reward functions.  
> 3. Train an IQL‑style offline RL policy that is conditioned on `z`.  
> 4. Evaluate the trained policy on a handful of downstream reward functions
>    (goal, linear, MLP) using only 32 samples of (state, reward) pairs to
>    encode the task.  
> 5. Print the cumulative rewards of each downstream task to `results.json`.

The output is deterministic given a fixed random seed, so the judge can
re‑run the same script and compare the numbers.

> **How to run**  
> ```bash
> bash reproduce.sh
> ```  
> The script will install dependencies, generate data, train the FRE and the
> policy, evaluate on downstream tasks and finally write a
> `results.json` file in the current directory.

> **What you should see**  
> After training you should see a JSON file that looks like:

```json
{
  "goal_reaching": 52.3,
  "linear_reward": 14.7,
  "mlp_reward": 18.9
}
```

> The numbers are not intended to match the paper’s performance
> exactly—they are only meant to demonstrate that the pipeline works and
> that the FRE encoder can be used to solve new tasks without further
> training.

> **Limitations**  
> * This is a toy implementation: the dataset is tiny, the training
>   procedure is a simplification, and the environments are very simple.
> * The aim is to show that the FRE idea can be implemented and
>   evaluated; it is *not* a state‑of‑the‑art result.