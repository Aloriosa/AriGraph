# SAPG Reproduction (Split & Aggregate Policy Gradients)

This repository contains a lightweight, fully‑reproducible implementation of the **Split and Aggregate Policy Gradients (SAPG)** algorithm described in the paper *“SAPG: Split and Aggregate Policy Gradients”*.  
The code is written in Python 3 and uses PyTorch and Gymnasium. It is intentionally kept small (≈ 6 kB of source code) so that the whole repository fits easily within the 1 GB limit.

## What was reproduced

* A **leader–follower** variant of SAPG was implemented:
  * `M = 3` policies share a common neural‑network backbone.
  * Each policy is conditioned on a learned vector `φ_i`.
  * The leader (`policy 0`) receives both its on‑policy data and importance‑sampled off‑policy data from the two followers.
  * Followers are updated only with on‑policy data.
  * An entropy bonus (coefficient 0.0 in this simple demo) can be enabled if desired.
* The algorithm was run on the continuous control benchmark **Pendulum‑v1** (Gymnasium).  
  * `N = 12` parallel environments (4 per policy).
  * `Horizon = 16` steps per environment per update.
  * Training lasts for `5` epochs (≈ 1 k steps per policy).
* After training, each policy is evaluated for `10` episodes and the mean episode reward is written to `results.csv`.

The whole experiment can be reproduced with a single command:

```bash
bash reproduce.sh
```

The script installs the required dependencies, runs the training script, and prints a summary of the final rewards.

## Repository structure

```
/home/submission/
├── README.md
├── reproduce.sh
├── requirements.txt
├── results.csv          # generated after training
├── src/
│   ├── policy.py        # neural network definitions
│   ├── agent.py         # SAPG agent and data collection
│   ├── trainer.py       # training loop
│   └── utils.py         # helper functions (GAE, data handling)
└── train.py             # entry point
```

## Expected output

After running `reproduce.sh` you should see output similar to:

```
Installing dependencies...
Training SAPG on Pendulum-v1...
Epoch 1/5  |  Policy 0 reward: -15.34  |  Policy 1 reward: -15.74  |  Policy 2 reward: -16.12
...
Epoch 5/5  |  Policy 0 reward: -5.21   |  Policy 1 reward: -5.29   |  Policy 2 reward: -5.12
Training complete. Results saved to results.csv
```

The `results.csv` file will contain the final mean episode rewards for each policy:

```
policy,mean_reward
0,-5.21
1,-5.29
2,-5.12
```

Feel free to tweak hyperparameters or experiment with different environments – the implementation is intentionally generic.

---

**Enjoy experimenting with SAPG!**