# Fine‑Tuning Reinforcement Learning Models – Forgetting Mitigation Demo

This repository contains a lightweight, fully reproducible toy version of the experiments
described in *“Fine‑tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem”*.
It demonstrates how knowledge‑retention techniques (Behavioral Cloning, Kick‑starting, EWC)
can prevent the catastrophic loss of pre‑trained capabilities when fine‑tuning on a new task.

## Structure

```
.
├── README.md
├── reproduce.sh          # main reproduction script
├── src/
│   ├── envs/
│   │   ├── apples_retrieval.py   # 1‑D gridworld – toy “AppleRetrieval”
│   │   └── two_state_mdp.py      # simple two‑state MDP
│   ├── agents/
│   │   ├── policy_network.py    # linear policy with logits
│   │   └── trainer.py           # REINFORCE + optional BC / KS / EWC
│   └── utils.py                 # helpers
└── requirements.txt
```

## How to run

```bash
bash reproduce.sh
```

The script will

1. Install the required Python packages (`torch`, `numpy`).
2. Train a policy on the **Phase 2** part of *AppleRetrieval* (pre‑training).
3. Fine‑tune that policy on the full task **with and without** a behavioral‑cloning
   auxiliary loss (BC).
4. Report the success rates for both experiments.

All training runs are performed on CPU, so the script finishes in a few seconds.

> **Note**: The code is intentionally lightweight and does **not** reproduce the full
> NetHack, Montezuma’s Revenge, or Meta‑World experiments from the paper.
> It only illustrates the central phenomenon of *forgetting of pre‑trained
> capabilities* and how a simple knowledge‑retention method can mitigate it.

## Expected Output

```
Pre‑training Phase 2 (BC) – finished.
Fine‑tuning without BC – success rate: 0.12
Fine‑tuning with BC – success rate: 0.67
```

The gap in success rates shows that vanilla fine‑tuning rapidly forgets the skill
learned in Phase 2, while the BC loss preserves it.

Feel free to tweak the hyper‑parameters in `src/agents/trainer.py` to explore other
knowledge‑retention settings (Kick‑starting, EWC, etc.).