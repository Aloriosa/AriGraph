#!/usr/bin/env bash
set -e

# ------------------------------------------------------------------
# 1. Install dependencies
# ------------------------------------------------------------------
echo "Installing dependencies..."
pip install -r requirements.txt

# ------------------------------------------------------------------
# 2. Create directories
# ------------------------------------------------------------------
mkdir -p checkpoints logs

# ------------------------------------------------------------------
# 3. Run training for each backbone and dataset
# ------------------------------------------------------------------
BACKBONES=("resnet18" "resnet50" "vit_b32")
DATASETS=("cifar10" "cifar100" "svhn" "gtsrb")

echo "Starting experiments..."
for backbone in "${BACKBONES[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "Training ${backbone} on ${dataset}"
    python train.py \
      --backbone $backbone \
      --dataset $dataset \
      --epochs 10 \
      --batch_size 128 \
      --lr 0.01 \
      --output_dir checkpoints/${backbone}_${dataset}
  done
done

# ------------------------------------------------------------------
# 4. Aggregate results
# ------------------------------------------------------------------
echo "Aggregating results..."
python aggregate_results.py > results.txt
echo "All experiments finished. Results written to results.txt"