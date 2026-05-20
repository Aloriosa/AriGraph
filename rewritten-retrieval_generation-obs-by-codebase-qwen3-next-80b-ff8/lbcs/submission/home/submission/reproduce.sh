#!/bin/bash
set -e

# Install dependencies
apt-get update && apt-get install -y python3 python3-pip python3-dev git wget
pip3 install torch torchvision numpy scikit-learn matplotlib flaml tqdm

# Create necessary directories
mkdir -p /home/submission/data
mkdir -p /home/submission/results
mkdir -p /home/submission/tmp

# Download and prepare datasets
python3 -c "
import torch
import torchvision
from torchvision import transforms
import numpy as np

# Download FashionMNIST
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(root='/home/submission/data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='/home/submission/data', train=False, download=True, transform=transform)

# Create noisy labels for FashionMNIST (30% symmetric noise)
np.random.seed(42)
num_classes = 10
num_samples = len(train_dataset.targets)
noisy_targets = np.array(train_dataset.targets)
noise_mask = np.random.random(num_samples) < 0.3
noise_labels = np.random.randint(0, num_classes, size=np.sum(noise_mask))
noisy_targets[noise_mask] = noise_labels
np.save('/home/submission/tmp/fashion_noisy_target.npy', noisy_targets)

# Download CIFAR-10
train_dataset_cifar = torchvision.datasets.CIFAR10(root='/home/submission/data', train=True, download=True, transform=transform)
test_dataset_cifar = torchvision.datasets.CIFAR10(root='/home/submission/data', train=False, download=True, transform=transform)

# Create noisy labels for CIFAR-10 (30% symmetric noise)
np.random.seed(42)
num_classes = 10
num_samples = len(train_dataset_cifar.targets)
noisy_targets_cifar = np.array(train_dataset_cifar.targets)
noise_mask = np.random.random(num_samples) < 0.3
noise_labels = np.random.randint(0, num_classes, size=np.sum(noise_mask))
noisy_targets_cifar[noise_mask] = noise_labels
np.save('/home/submission/tmp/cifar10_noisy_target.npy', noisy_targets_cifar)

# Download SVHN
transform_svhn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
])
train_dataset_svhn = torchvision.datasets.SVHN(root='/home/submission/data', split='train', download=True, transform=transform_svhn)
test_dataset_svhn = torchvision.datasets.SVHN(root='/home/submission/data', split='test', download=True, transform=transform_svhn)

# Create noisy labels for SVHN (30% symmetric noise)
np.random.seed(42)
num_classes = 10
num_samples = len(train_dataset_svhn.labels)
noisy_targets_svhn = np.array(train_dataset_svhn.labels)
noise_mask = np.random.random(num_samples) < 0.3
noise_labels = np.random.randint(0, num_classes, size=np.sum(noise_mask))
noisy_targets_svhn[noise_mask] = noise_labels
np.save('/home/submission/tmp/svhn_noisy_target.npy', noisy_targets_svhn)

print('Datasets downloaded and noisy labels generated')
"

# Run LBCS on FashionMNIST with 30% symmetric label noise
python3 lbcs.py --dataset fashion_mnist --noise_rate 0.3 --coreset_size 1000 --tolerance 15% --train_epochs 100 --save_dir /home/submission/results/fashion_mnist

# Run LBCS on CIFAR-10 with 30% symmetric label noise
python3 lbcs.py --dataset cifar10 --noise_rate 0.3 --coreset_size 4000 --tolerance 15% --train_epochs 100 --save_dir /home/submission/results/cifar10

# Run LBCS on SVHN with 30% symmetric label noise
python3 lbcs.py --dataset svhn --noise_rate 0.3 --coreset_size 1000 --tolerance 15% --train_epochs 100 --save_dir /home/submission/results/svhn

# Run comparison with baselines on FashionMNIST
python3 baseline_comparison.py --dataset fashion_mnist --noise_rate 0.3 --coreset_size 1000 --tolerance 15% --save_dir /home/submission/results/baselines

# Generate results summary
python3 summarize_results.py --results_dir /home/submission/results --output /home/submission/results/summary.csv

echo "Reproduction completed. Results saved to /home/submission/results/"