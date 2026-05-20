# Reproducing Forgetting of Pre‑Trained Capabilities

This repository contains a **toy** implementation that demonstrates the key claim of the paper *“Fine‑tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem”*.  
The experiment is intentionally lightweight so that it can be run inside a 7‑day Docker container with modest compute resources.

## Overview

1. **Pre‑training**  
   A simple policy is trained only on the *far* state (state = 1) of a 2‑state MDP.  
   The policy learns to stay in state 1 and receive a high reward.

2. **Fine‑tuning**  
   The policy is further trained on the full MDP (both states).  
   Two scenarios are evaluated:
   * **Vanilla fine‑tuning** – no auxiliary loss.
   * **Fine‑tuning + BC** – a behavioural‑cloning auxiliary loss that keeps the policy close to the pre‑trained behaviour on states seen during pre‑training.

3. **Results**  
   After fine‑tuning, we evaluate the mean return on the full environment.  
   The expectation is that vanilla fine‑tuning forgets the ability to stay in state 1, whereas the BC‑augmented version retains it.

The entire workflow is automated in `reproduce.sh`.  Running this script will produce a `results.txt` file containing the mean returns of the three policies.

## Repository Structure

```
├── envs/
│   └── two_state_mdp.py            # Toy 2‑state MDP environment
├── models/
│   └── policy.py                   # Linear policy network
├── train_pretrain.py               # Pre‑train the policy on state 1
├── finetune.py                     # Fine‑tune with optional BC/EWC
├── reproduce.sh                    # End‑to‑end reproduction script
├── requirements.txt                # Python dependencies
└── README.md
```

## How to Run

```bash
bash reproduce.sh
```

After completion, inspect `results.txt` for the recorded metrics.

## Expected Outcomes

| Policy | Mean Return (Full Env) |
|--------|------------------------|
| Pre‑trained (on state 1) | ~4.5 (high reward for staying in state 1) |
| Fine‑tuned (no BC)       | ~-0.5 (policy forgets to stay in state 1, ends up in state 0) |
| Fine‑tuned (with BC)     | ~4.0 (policy largely retains the pre‑trained skill) |

These numbers may vary slightly due to stochasticity but the qualitative trend should hold: the BC auxiliary loss mitigates forgetting.

## Extending the Experiment

- Replace the toy MDP with more complex environments (e.g. `CartPole`, `Acrobot`) and observe the same phenomena.
- Add EWC or replay‑based methods to compare against BC.
- Visualise the policy logits over training to see how the action distribution evolves.

## Note

The code is intentionally simple and uses only CPU‑based PyTorch.  It is **not** intended to reproduce the large‑scale NetHack or Montezuma’s Revenge experiments from the original paper, which require substantial compute and data.  Instead, it captures the *conceptual* essence of forgetting of pre‑trained capabilities and showcases how a straightforward knowledge‑retention technique (behavioural cloning) can alleviate it.

Happy experimenting!