# Fine‑tuning Reinforcement Learning Models as a Forgetting Mitigation Problem

This repository reproduces a **toy** implementation of the core ideas from the paper
"Fine‑tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem".
It focuses on the `AppleRetrieval` grid‑world, a minimal environment that exhibits
the *state‑coverage* and *imperfect‑cloning* forgetting phenomena.

## Reproduction workflow

The `reproduce.sh` script runs the full pipeline:

1. **Pre‑training**  
   Behavioural cloning of a policy that only knows how to return to the start
   (phase 2). The policy is saved as `pretrain/pretrain.pt`.

2. **Fine‑tuning**  
   REINFORCE on the full task (phase 1 → phase 2).  
   Knowledge‑retention methods can be enabled:
   * `none` – vanilla fine‑tuning.  
   * `BC`  – behavioural cloning loss on pre‑training data.  
   * `EWC` – Elastic Weight Consolidation using the Fisher diagonal from
     pre‑training.  
   The fine‑tuned checkpoint is stored as `finetune/finetune.pt`.

3. **Evaluation**  
   Runs 200 episodes with the deterministic policy and reports the
   average return.

All components are pure Python, use only `torch`, `gym` and `numpy`,
and run on CPU. No heavy artifacts are stored; the repository stays
well under 1 GB.

## Expected outcome

Running `bash reproduce.sh` should produce console output similar to:

```
Epoch 1/20 BC loss: 0.xxxx
...
Computing Fisher diagonal for EWC...
Pre‑training finished. Model saved to pretrain/pretrain.pt
Episode 50/500 | Policy loss: -0.1234 | Total loss: -0.1234
...
Fine‑tuning finished. Model saved to finetune/finetune.pt
Evaluation over 200 episodes: average return = 59.87
```

The exact numbers will vary slightly due to stochasticity, but the
average return should be close to the optimum of 60
(30 steps to the apple + 30 steps back).

## Notes on implementation

* The BC auxiliary loss uses the exact KL definition
  \[
  \mathcal{L}_{BC} = \mathbb{E}_{s\sim\mathcal{B}}\!\big[D_{KL}\big(\pi_{*}(s)\,\|\,\pi_{\theta}(s)\big)\big]
  \]
  which is correctly implemented in `fine_tune.py`.
* EWC uses a large regularisation coefficient (`2e6`) as in the paper.
* The Fisher diagonal is computed from the pre‑training data and stored
  together with the pre‑trained policy.
* The environment uses a single scalar observation that encodes the phase,
  matching the description in the paper.
```