#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
python3 -m pip install --quiet -r requirements.txt

# Create results directory
mkdir -p results

# 1. Pre‑train
echo "=== Pre‑training (Phase 2) ==="
python3 train.py --mode pretrain --env_name AppleRetrieval_pretrain --seed 42

# 2. Fine‑tune (vanilla)
echo "=== Fine‑tune (vanilla) ==="
python3 train.py --mode finetune --method vanilla --env_name AppleRetrieval_finetune --seed 43
mv policy_finetune_vanilla.pt results/policy_finetune_vanilla.pt
python3 train.py --mode eval --policy results/policy_finetune_vanilla.pt --env_name AppleRetrieval_finetune --seed 44
mv results.txt results/vanilla.txt

# 3. Fine‑tune with BC
echo "=== Fine‑tune (BC) ==="
python3 train.py --mode finetune --method bc --env_name AppleRetrieval_finetune --seed 45
mv policy_finetune_bc.pt results/policy_finetune_bc.pt
python3 train.py --mode eval --policy results/policy_finetune_bc.pt --env_name AppleRetrieval_finetune --seed 46
mv results.txt results/BC.txt

# 4. Fine‑tune with EWC
echo "=== Fine‑tune (EWC) ==="
python3 train.py --mode finetune --method ewc --env_name AppleRetrieval_finetune --seed 47
mv policy_finetune_ewc.pt results/policy_finetune_ewc.pt
python3 train.py --mode eval --policy results/policy_finetune_ewc.pt --env_name AppleRetrieval_finetune --seed 48
mv results.txt results/EWC.txt

# 5. Fine‑tune with KS
echo "=== Fine‑tune (KS) ==="
python3 train.py --mode finetune --method ks --env_name AppleRetrieval_finetune --seed 49
mv policy_finetune_ks.pt results/policy_finetune_ks.pt
python3 train.py --mode eval --policy results/policy_finetune_ks.pt --env_name AppleRetrieval_finetune --seed 50
mv results.txt results/KS.txt

# 6. Fine‑tune with EM
echo "=== Fine‑tune (EM) ==="
python3 train.py --mode finetune --method em --env_name AppleRetrieval_finetune --seed 51
mv policy_finetune_em.pt results/policy_finetune_em.pt
python3 train.py --mode eval --policy results/policy_finetune_em.pt --env_name AppleRetrieval_finetune --seed 52
mv results.txt results/EM.txt

# 7. Evaluate on FAR states (start at M)
echo "=== Evaluation on FAR states (start at M) ==="
for m in vanilla bc ewc ks em; do
    python3 train.py --mode eval_far --policy results/policy_finetune_${m}.pt --env_name AppleRetrieval_finetune --seed 60
    mv results.txt results/${m}_far.txt
done

echo "=== Reproduction finished ==="
echo "Results are stored in the 'results/' directory."