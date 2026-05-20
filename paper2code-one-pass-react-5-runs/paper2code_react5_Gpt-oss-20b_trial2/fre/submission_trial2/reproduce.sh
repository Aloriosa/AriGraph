#!/usr/bin/env bash
set -e

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Training Functional Reward Encoder (FRE) ==="
python train_fre.py --env halfcheetah-medium-expert-v2 --output fre_checkpoint --steps 150000 --seed 42

echo "=== Training latent-conditioned policy (IQL) ==="
python train_policy.py --fre_ckpt fre_checkpoint/fre_checkpoint.pt \
                       --policy_ckpt policy_checkpoint \
                       --env halfcheetah-medium-expert-v2 \
                       --steps 850000 \
                       --seed 42

echo "=== Running zero-shot evaluation ==="
python evaluate_policy.py --policy_ckpt policy_checkpoint/policy_checkpoint.pt \
                          --fre_ckpt fre_checkpoint/fre_checkpoint.pt \
                          --env halfcheetah-medium-expert-v2 \
                          --episodes 5 \
                          --seed 42

echo "=== Reproduction finished. Check 'policy_evaluation.json' for results ==="