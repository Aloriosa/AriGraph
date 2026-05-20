#!/usr/bin/env bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading CIFAR-100 dataset..."
python -c "import torchvision; print('Dataset downloaded')"

echo "Running SEMA training..."
python sema_train.py --config config.yaml