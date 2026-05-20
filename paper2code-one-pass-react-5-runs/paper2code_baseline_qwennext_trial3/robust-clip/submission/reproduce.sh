#!/bin/bash
# Reproduce the results from the paper "Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models"

set -e  # Exit on any error

echo "=== Reproducing Robust CLIP: Unsupervised Adversarial Fine-Tuning for Robust LVLMs ==="

# Install required packages
echo "Installing required packages..."
apt-get update
apt-get install -y python3 python3-pip git

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install torch torchvision numpy matplotlib scikit-learn

# Create necessary directories
echo "Creating project structure..."
mkdir -p /home/submission/src
mkdir -p /home/submission/data
mkdir -p /home/submission/results

# Copy source code to the submission directory
echo "Copying source code...
cp /home/submission/src/*.py /home/submission/src/

# Download and prepare the dataset
echo "Downloading and preparing the CIFAR-10 dataset..."
python3 -c "
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='/home/submission/data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='/home/submission/data', train=False, download=True, transform=transform)

# Save the datasets
torch.save(trainset, '/home/submission/data/cifar10_train.pt')
torch.save(testset, '/home/submission/data/cifar10_test.pt')

print('Dataset downloaded and saved successfully.')

# Run the FARE training script
echo "Running FARE training script..."
python3 /home/submission/src/train_fare.py \
  --dataset /home/submission/data/cifar10_train.pt \
  --epochs 5 \
  --batch_size 32 \
  --epsilon 2/255 \
  --output /home/submission/results/robust_clip_model.pth

# Run the evaluation script
echo "Running evaluation script...
python3 /home/submission/src/evaluate.py \
  --model /home/submission/results/robust_clip_model.pth \
  --dataset /home/submission/data/cifar10_test.pt \
  --output /home/submission/results/evaluation_results.json

# Generate plots
echo "Generating plots...
python3 /home/submission/src/plot_results.py \
  --input /home/submission/results/evaluation_results.json \
  --output /home/submission/results/robustness_plot.png

echo "Reproduction completed successfully!"