# CompoNet Reproduction

This repository contains a lightweight, self‑contained implementation of the **CompoNet** architecture described in the paper *“Self‑Composing Policies for Scalable Continual Reinforcement Learning”*.  The goal is to provide a minimal but executable reproduction pipeline that:

1. Instantiates the core CompoNet module (self‑composing policy modules with attention heads).
2. Trains a simple SAC agent on the Meta‑World task sequence.
3. Trains a simple PPO agent on the ALE SpaceInvaders and Freeway task sequences.
4. Logs the final success rate for each task and writes a JSON summary of the results.

> **NOTE** – This implementation is intentionally lightweight.  It focuses on reproducing the *architecture* and the *training pipeline* rather than the exact numerical results from the paper (which would require several days of GPU time).  The code is fully self‑contained, requires only the packages listed in `requirements.txt`, and can be executed with the `reproduce.sh` script on a machine with an NVIDIA GPU.

## Reproduction

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Train the agents on the three task sequences.
3. Store the results in `results.json`.

The final file `results.json` contains the averaged success rates for each method and task.  Feel free to inspect the logs for more detail.

## Repository Structure

```
├── requirements.txt            # pip dependencies
├── README.md                   # this file
├── reproduce.sh                # main reproduction script
├── compo.py                    # CompoNet architecture
├── train_meta_world.py         # SAC training on Meta‑World
├── train_ale.py                # PPO training on ALE (SpaceInvaders & Freeway)
├── utils.py                    # helper utilities
└── results.json                # automatically generated after training
```

---

If you encounter any issues, open an issue or submit a pull‑request.  Happy experimenting!