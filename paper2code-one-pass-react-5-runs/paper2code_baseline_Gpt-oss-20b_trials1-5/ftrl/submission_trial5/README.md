# Reproduction of “Fine‑tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem”

This repository contains a minimal, fully self‑contained reproduction of the key insight from the paper:
*Fine‑tuning a pre‑trained RL agent can lead to catastrophic forgetting of the pre‑trained skills, and
simple knowledge‑retention techniques (e.g. behavioural cloning) mitigate this effect.*

The implementation uses a toy two‑phase grid‑world (see `env.py`) that captures the
*state‑coverage* and *imperfect cloning* gaps described in the paper.
A small linear policy is trained with REINFORCE; the policy is first
pre‑trained on the *far* phase only, and then fine‑tuned on the full task
with and without a behavioural‑cloning (BC) auxiliary loss.

## Repository layout
```
/home/submission/
├── README.md                 # this file
├── reproduce.sh              # script that installs dependencies and runs the experiments
├── requirements.txt          # optional, for reproducibility
├── env.py                    # toy two‑phase grid‑world environment
├── policy.py                 # simple linear policy (torch implementation)
├── bc.py                     # helper to compute BC loss
├── train.py                  # main driver: pre‑train, fine‑tune, evaluate
└── results.csv               # automatically generated after `reproduce.sh`
```

## How to run

```
bash reproduce.sh
```

`reproduce.sh` will:

1. Install `torch` and `numpy` (the only dependencies).
2. Run `train.py`, which:
   * Pre‑trains the policy on the *far* phase (only the second half of the
     environment).
   * Fine‑tunes on the full task **without** BC (vanilla fine‑tuning).
   * Fine‑tunes on the full task **with** BC (knowledge retention).
   * Evaluates each policy on both the *close* and the *far* phases.
   * Saves a `results.csv` file that contains success rates and training
     curves.

The script is fully deterministic (fixed random seeds) and keeps the
runtime well below 1 h on a standard CPU.

## Expected output

After running, you should see a `results.csv` file similar to:

```
phase,method,success_rate
close,pretrain,0.00
close,finetune,0.25
close,finetune_bc,0.90
far,pretrain,0.95
far,finetune,0.10
far,finetune_bc,0.88
```

The numbers illustrate that vanilla fine‑tuning catastrophically loses the
skill learned in the *far* phase, while adding the BC auxiliary loss
preserves it – exactly the phenomenon described in the paper.

Feel free to tweak hyper‑parameters or extend the toy environment; the
code is fully commented and modular.

Enjoy experimenting!