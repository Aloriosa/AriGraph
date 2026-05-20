#!/usr/bin/env bash
set -e

# Update package list and install python3.10
apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev

# Create a virtual environment and activate it
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required Python packages
pip install --quiet torch torchvision timm tqdm

# Create a directory for logs
mkdir -p logs

# List of supported datasets and backbones
datasets=("CIFAR10" "CIFAR100" "SVHN" "GTSRB" "Flowers102" "DTD" "UCF101" "Food101" "EuroSAT" "OxfordPets" "SUN397")
backbones=("resnet18" "vitb32")

# Run training for each dataset and backbone combination
echo "=== Training started ===" > results.txt
for dataset in "${datasets[@]}"; do
  for backbone in "${backbones[@]}"; do
    echo "Training ${dataset} with ${backbone}..." | tee -a results.txt
    python train.py \
      --mode smm \
      --dataset "$dataset" \
      --backbone "$backbone" \
      --epochs 20 \
      --batch-size 256 \
      --output "logs/${dataset}_${backbone}_smm.txt" >> results.txt 2>&1
    echo "Baseline run..." | tee -a results.txt
    python train.py \
      --mode baseline \
      --dataset "$dataset" \
      --backbone "$backbone" \
      --epochs 20 \
      --batch-size 256 \
      --output "logs/${dataset}_${backbone}_baseline.txt" >> results.txt 2>&1
    echo "-----" >> results.txt
  done
done

echo "=== Training finished ===" >> results.txt
echo "Reproduction finished. Results written to results.txt"