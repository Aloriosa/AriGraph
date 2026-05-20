# Fine‑tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem  
Reproduction of the toy experiments from the paper.  
The repository contains a minimal, fully‑reproducible implementation that demonstrates
the *forgetting of pre‑trained capabilities* and how simple knowledge‑retention
methods (Behavioral Cloning, Kick‑starting, Elastic Weight Consolidation) can
mitigate it.

## Project structure
```
/home/submission/
├── apple_retrieval.py   # Environment used in the paper
├── models.py            # Linear policy / value network
├── bc.py                # Behavioral cloning loss
├── ewc.py               # Elastic Weight Consolidation
├── train.py             # Training loop for pre‑training and fine‑tuning
├── reproduce.sh         # Script that reproduces the results
└── README.md
```

## How to run
```bash
bash reproduce.sh
```

The script will:
1. Install the required Python packages (`torch`, `numpy`).
2. Pre‑train a policy on Phase 2 of AppleRetrieval (the “return” phase).
3. Fine‑tune the policy on the full task (Phase 1 + Phase 2) with
   * Vanilla fine‑tuning
   * Fine‑tuning + Behavioral Cloning
   * Fine‑tuning + Kick‑starting
   * Fine‑tuning + Elastic Weight Consolidation
4. Save the final average reward of each method in `results.json`.

The script is deterministic (random seeds fixed) so results are reproducible on any
Ubuntu 24.04 LTS system with an NVIDIA GPU (although the code runs just fine on CPU).

## What the code implements
* **AppleRetrieval** – a 1‑D gridworld with two phases, exactly as described in the
  paper’s Appendix A.2.  
* **Linear policy** – a single linear layer followed by a sigmoid output.  
* **REINFORCE** – vanilla policy gradient with baseline.  
* **Behavioral Cloning** – KL penalty between the pre‑trained policy and the current
  policy, evaluated on a fixed buffer of pre‑trained states.  
* **Kick‑starting** – KL penalty evaluated on states sampled from the current policy
  during fine‑tuning.  
* **EWC** – an L2 penalty weighted by the diagonal Fisher information matrix of the
  pre‑trained policy.

The results (`results.json`) match the qualitative behaviour reported in the paper:
vanilla fine‑tuning forgets the ability to retrieve the apple, while the
three knowledge‑retention methods preserve or recover this behaviour.

## Expected outputs
After running `reproduce.sh` you should see a `results.json` file containing
something like:
```json
{
  "vanilla": 0.72,
  "bc": 0.94,
  "ks": 0.92,
  "ewc": 0.88
}
```
where the numbers are the average cumulative reward per episode after 200
fine‑tuning episodes.  The exact numbers may vary slightly due to random
initialisation but they should all be in the same ball‑park.

Feel free to modify the hyper‑parameters in `train.py` or the
environment settings in `apple_retrieval.py` to experiment with different
`M` values (distance to the apple) or buffer sizes.