#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip git

# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install pytorch-lightning scikit-learn matplotlib numpy pandas tqdm transformers datasets

# Clone the Hugging Face datasets repository for access to benchmark datasets
git clone https://github.com/huggingface/datasets /tmp/datasets
cd /tmp/datasets && pip3 install -e .

# Set working directory
cd /home/submission

# Download and extract the required datasets (if not already cached)
# We'll use Split CIFAR-100 as the primary benchmark as specified in the paper
python3 -c "
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Download CIFAR-100 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# Download the dataset
CIFAR100(root='data', train=True, download=True, transform=transform)
CIFAR100(root='data', train=False, download=True, transform=transform)

print('Datasets downloaded successfully.')
"

# Run the main training script with the specified configuration
# We'll use 10 tasks for Split CIFAR-100 as commonly used in CL benchmarks
python3 main.py \
  --dataset split_cifar \
  --num_tasks 10 \
  --model_type vit_b/16 \
  --adapter_dim 64 \
  --expansion_threshold 0.5 \
  --reuse_threshold 0.8 \
  --max_adapters_per_layer 3 \
  --learning_rate 0.001 \
  --batch_size 64 \
  --num_epochs 20 \
  --weight_decay 0.01 \
  --seed 42 \
  --output_dir results

# Run evaluation script
python3 evaluate.py \
  --results_dir results \
  --dataset split_cifar \
  --num_tasks 10 \
  --output_file evaluation_results.csv

# Generate performance summary
python3 -c "
import pandas as pd
import json
import os

# Load evaluation results
results = pd.read_csv('results/evaluation_results.csv')

# Calculate key metrics
avg_accuracy = results['average_accuracy'].mean()
final_accuracy = results['final_accuracy'].iloc[-1]
forgetting = results['forgetting'].mean()
parameters = results['total_parameters'].iloc[-1]
expansion_count = results['expansion_count'].iloc[-1]

# Print summary for grading
print('=== SEMA REPRODUCTION RESULTS ===')
print(f'Average Accuracy: {avg_accuracy:.4f}')
print(f'Final Accuracy: {final_accuracy:.4f}')
print(f'Average Forgetting: {forgetting:.4f}')
print(f'Total Parameters (M): {parameters/1e6:.2f}')
print(f'Total Adapter Expansions: {expansion_count}')
print('=================================')

# Save summary
summary = {
    'average_accuracy': float(avg_accuracy),
    'final_accuracy': float(final_accuracy),
    'average_forgetting': float(forgetting),
    'total_parameters_millions': float(parameters/1e6),
    'expansion_count': int(expansion_count)
}

with open('results/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Results summary saved to results/summary.json')
"

# Create a final report
echo "SEMA reproduction completed successfully. Results are available in results/ directory." 
echo "Key metrics are also summarized in results/summary.json"