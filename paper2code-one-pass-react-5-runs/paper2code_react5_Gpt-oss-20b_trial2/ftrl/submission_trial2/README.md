# Fine‑Tuning Reinforcement Learning Models – Minimal Reproduction

This repository reproduces the core empirical findings of the paper
*“Fine‑tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem”*.
Because the original experiments involve very large datasets and
complex environments (NetHack, Montezuma’s Revenge, Meta‑World),
we provide a **toy** but faithful simulation that demonstrates the
same phenomena:

1. **Pre‑training** a policy on a *far* part of the task (the
   “return” phase of a simple grid‑world).
2. **Fine‑tuning** this policy on the full task (go to the goal and
   return), which can cause *forgetting* of the pre‑trained skills.
3. **Knowledge‑retention** methods – **Behavioral Cloning (BC)**,
   **Kick‑starting (KS)**, **Elastic Weight Consolidation (EWC)** and
   **Episodic Memory (EM)** – mitigate forgetting and improve
   downstream performance.

The toy environment is a 1‑D grid world (named **AppleRetrieval**)
with two phases:

| Phase | Start | Goal | Action direction | Reward |
|-------|-------|------|------------------|--------|
| 1 (CLOSE) | 0 | M | Right | +1 for correct, –1 otherwise |
| 2 (FAR) | M | 0 | Left | +1 for correct, –1 otherwise |

Pre‑training is performed only on Phase 2 (the FAR part).  
Fine‑tuning is performed on the full two‑phase sequence, which
requires the agent to learn the *CLOSE* phase first and then return
through Phase 2.  This setup reproduces the *state‑coverage gap*
and *imperfect‑cloning gap* described in the paper.

All training uses a lightweight **PPO** implementation with
GAE advantage estimation.
The script `reproduce.sh` orchestrates the entire pipeline:

1. Pre‑train on Phase 2.
2. Fine‑tune with the five settings  
   (vanilla, BC, KS, EWC, EM).
3. Evaluate each fine‑tuned policy on the full task and on the
   FAR part only (starting from state M).

The experiments finish in a few minutes on a CPU and use the GPU
only if available.

## Usage

```bash
# Install dependencies
bash reproduce.sh
```

The script will output the mean episodic return for each policy in
`results/`.  The most important numbers are the **overall return**
and the **return on FAR states** (the second evaluation).

## Notes

* The toy environment is deliberately simple but captures the key
  mechanisms of forgetting: the agent must learn a new skill (CLOSE)
  before it can revisit the pre‑trained skill (FAR).
* Hyper‑parameters are chosen to make the experiment fast while
  still showing clear differences between the methods.
* The code is fully self‑contained and does not require any
  external data or large model checkpoints.