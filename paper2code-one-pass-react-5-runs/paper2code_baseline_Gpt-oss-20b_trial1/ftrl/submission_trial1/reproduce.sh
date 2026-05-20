#!/usr/bin/env bash
set -euo pipefail
# The script reproduces the toy forgetting experiment.
# It trains a pre‑trained policy on only state 1, then fine‑tunes
# on the full two‑state MDP with and without a behavioural‑cloning
# auxiliary loss (BC).  Results are written to results.txt.

# 1. Install dependencies
pip install -q -r requirements.txt

# 2. Pre‑train a policy on state 1 only
python train_pretrain.py --epochs 2000 --output pretrain.pt

# 3. Fine‑tune without any knowledge‑retention
python finetune.py \
    --pretrain pretrain.pt \
    --output finetune_no_bc.pt \
    --epochs 2000

# 4. Fine‑tune with behavioural cloning (BC)
python finetune.py \
    --pretrain pretrain.pt \
    --output finetune_bc.pt \
    --epochs 2000 \
    --bc

# 5. Collect results
echo "=== Results ===" > results.txt
echo "Pre‑trained policy mean return (full env):" >> results.txt
python -c "from models.policy import LinearPolicy; import torch; from envs.two_state_mdp import TwoStateMDP; from finetune import evaluate; p=LinearPolicy(); p.load_state_dict(torch.load('pretrain.pt')); print(evaluate(p, TwoStateMDP()))" >> results.txt

echo "" >> results.txt
echo "Fine‑tuned (no BC) mean return:" >> results.txt
python -c "from models.policy import LinearPolicy; import torch; from envs.two_state_mdp import TwoStateMDP; from finetune import evaluate; p=LinearPolicy(); p.load_state_dict(torch.load('finetune_no_bc.pt')); print(evaluate(p, TwoStateMDP()))" >> results.txt

echo "" >> results.txt
echo "Fine‑tuned (with BC) mean return:" >> results.txt
python -c "from models.policy import LinearPolicy; import torch; from envs.two_state_mdp import TwoStateMDP; from finetune import evaluate; p=LinearPolicy(); p.load_state_dict(torch.load('finetune_bc.pt')); print(evaluate(p, TwoStateMDP()))" >> results.txt

echo "Reproduction finished. Results written to results.txt"