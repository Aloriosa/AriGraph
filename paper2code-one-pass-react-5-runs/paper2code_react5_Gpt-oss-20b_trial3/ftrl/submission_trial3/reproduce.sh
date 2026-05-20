#!/usr/bin/env bash
set -euo pipefail

# Create results directory
mkdir -p results

# Install dependencies
pip install -q -r requirements.txt

# 1. Pre‑train on the far part of AppleRetrieval
echo "=== Pre‑training ==="
python train.py --pretrain --max_steps 20000 > results/pretrain.log

# 2. Fine‑tune on the full AppleRetrieval with different knowledge‑retention methods
methods=(vanilla ewc bc ks em)
for m in "${methods[@]}"; do
  echo "=== Fine‑tuning with method: $m ==="
  python train.py --method "$m" --max_steps 20000 > "results/fine_tune_${m}.log"
done

# 3. Summarise results
echo "=== Summary ==="
echo "Method,AvgReturn,SuccessRate" > results/summary.csv
for m in "${methods[@]}"; do
  avg=$(grep "Average return over" "results/fine_tune_${m}.log" | awk '{print $6}')
  succ=$(grep "Success rate:" "results/fine_tune_${m}.log" | awk '{print $4}')
  echo "$m,$avg,$succ" >> results/summary.csv
done

echo "All experiments finished. Results are in the 'results/' directory."